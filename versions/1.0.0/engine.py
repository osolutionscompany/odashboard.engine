"""
Odashboard Engine - Version 1.0.0
This file contains all the processing logic for dashboard visualizations.
"""
import json
import logging
import itertools
from datetime import datetime, date, timedelta, time
import calendar
import re
from dateutil.relativedelta import relativedelta

_logger = logging.getLogger(__name__)


def get_models(env):
    """
    Return a list of models relevant for analytics, automatically filtering out technical models

    Args:
        env: Odoo environment from the request
        
    Returns:
        List of analytically relevant models with name and model attributes
    """
    try:
        # Create domain to filter models directly in the search
        # 1. Must be non-transient
        domain = [('transient', '=', False)]

        # 2. Exclude technical models using NOT LIKE conditions
        technical_prefixes = ['ir.', 'base.', 'bus.', 'base_import.',
                            'web.', 'mail.', 'auth.', 'report.',
                            'resource.', 'wizard.']
        for prefix in technical_prefixes:
            domain.append(('model', 'not like', f'{prefix}%'))

        # Models starting with underscore
        domain.append(('model', 'not like', '\_%'))

        # Execute the optimized search
        model_obj = env['ir.model'].sudo()
        models = model_obj.search(domain)

        _logger.info("Found %s analytical models", len(models))

        # Format the response with the already filtered models
        model_list = [{
            'name': model.name,
            'model': model.model,
        } for model in models]

        return {'success': True, 'data': model_list}

    except Exception as e:
        _logger.error("Error in get_models: %s", str(e))
        return {'success': False, 'error': str(e)}


def get_model_fields(model_name, env):
    """
    Retrieve information about the fields of a specific Odoo model.

    Args:
        model_name: Name of the Odoo model (example: 'sale.order')
        env: Odoo environment from the request
        
    Returns:
        Dictionary with information about the model's fields
    """
    try:
        # Check if the model exists
        if model_name not in env:
            return {'success': False, 'error': f"Model '{model_name}' not found"}

        # Get field information
        model_obj = env[model_name].sudo()
        
        # Get fields from the model
        fields_data = model_obj.fields_get()
        
        # Fields to exclude
        excluded_field_types = ['binary', 'one2many', 'many2many', 'text']  # Binary fields like images in base64
        excluded_field_names = [
            '__last_update',
            'write_date', 'write_uid', 'create_uid',
        ]

        # Fields prefixed with these strings will be excluded
        excluded_prefixes = ['message_', 'activity_', 'has_', 'is_', 'x_studio_']
        
        # Get fields info
        fields_info = []
        
        for field_name, field_data in fields_data.items():
            field_type = field_data.get('type', 'unknown')

            # Skip fields that match our exclusion criteria
            if (field_type in excluded_field_types or
                field_name in excluded_field_names or
                any(field_name.startswith(prefix) for prefix in excluded_prefixes)):
                continue

            # Check if it's a computed field that's not stored
            field_obj = model_obj._fields.get(field_name)
            if field_obj and field_obj.compute and not field_obj.store:
                _logger.debug("Skipping non-stored computed field: %s", field_name)
                continue

            # Create field info object for response
            field_info = {
                'field': field_name,
                'name': field_data.get('string', field_name),
                'type': field_type,
                'label': field_data.get('string', field_name),
                'value': field_name,
                'search': f"{field_name} {field_data.get('string', field_name)}"
            }

            # Add selection options if field is a selection
            if field_data.get('type') == 'selection' and 'selection' in field_data:
                field_info['selection'] = [
                    {'value': value, 'label': label}
                    for value, label in field_data['selection']
                ]

            fields_info.append(field_info)

        # Sort fields by name for better readability
        fields_info.sort(key=lambda x: x['name'])

        return {'success': True, 'data': fields_info}

    except Exception as e:
        _logger.error("Error in get_model_fields: %s", str(e))
        return {'success': False, 'error': str(e)}


def _build_odash_domain(group_by_values):
    """
    Build odash.domain for a specific data point based on groupby values.
    Returns only the specific domain for this data point, not including the base domain.
    
    Args:
        group_by_values: Dictionary of field names and their values from the data point
        
    Returns:
        List of domain criteria (e.g. [[field, operator, value], ...]])
    """
    domain = []
    
    for field, value in group_by_values.items():
        # Skip count field
        if field == '__count':
            continue
            
        # Handle standard date fields that might need interval processing
        if field in ['date', 'create_date', 'write_date'] or field.endswith('_date'):
            # Format "DD MMM YYYY" (ex: "11 Apr 2025")
            if isinstance(value, str) and re.match(r'\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}', value):
                try:
                    # Parse the date string
                    parts = value.split(' ')
                    if len(parts) == 3:
                        day = int(parts[0])
                        month_str = parts[1]
                        year = int(parts[2])
                        
                        # Map month name to number
                        month_map = {
                            'Jan': 1, 'January': 1, 'Feb': 2, 'February': 2,
                            'Mar': 3, 'March': 3, 'Apr': 4, 'April': 4,
                            'May': 5, 'Jun': 6, 'June': 6, 'Jul': 7, 'July': 7,
                            'Aug': 8, 'August': 8, 'Sep': 9, 'Sept': 9, 'September': 9,
                            'Oct': 10, 'October': 10, 'Nov': 11, 'November': 11,
                            'Dec': 12, 'December': 12
                        }
                        
                        if month_str in month_map:
                            month = month_map[month_str]
                            
                            # Create start and end of day for the date
                            start_datetime = datetime(year, month, day, 0, 0, 0)
                            end_datetime = datetime(year, month, day, 23, 59, 59)
                            
                            # Add the date range to the domain
                            domain.append([field, '>=', start_datetime.isoformat()])
                            domain.append([field, '<=', end_datetime.isoformat()])
                            continue
                except Exception as e:
                    _logger.error("Error parsing date in domain: %s - %s", value, str(e))
                    # Fall through to default handling
        
        # Handle interval notation in field name (e.g., create_date:month)
        if ':' in field:
            base_field, interval = field.split(':')
            
            # Handle week pattern specifically
            if isinstance(value, str) and re.match(r'W\d{1,2}\s+\d{4}', value):
                # Handle week format by getting date range
                start_date, end_date = _parse_date_from_string(value, return_range=True)
                domain.append([base_field, '>=', start_date.isoformat()])
                domain.append([base_field, '<=', end_date.isoformat()])
                continue
                
            # Handle date intervals
            if interval in ('day', 'week', 'month', 'quarter', 'year'):
                if interval == 'month' and re.match(r'\d{4}-\d{2}', str(value)):
                    year, month = str(value).split('-')
                    start_date = date(int(year), int(month), 1)
                    end_date = date(int(year), int(month), calendar.monthrange(int(year), int(month))[1])
                    domain.append([base_field, '>=', start_date.isoformat()])
                    domain.append([base_field, '<=', end_date.isoformat()])
                    continue
                elif interval == 'day' and isinstance(value, str):
                    # Try to parse day format and create a range
                    try:
                        # Get date object using our extract_date function logic
                        date_formats = ['%d %b %Y', '%Y-%m-%d']
                        date_obj = None
                        
                        for fmt in date_formats:
                            try:
                                date_obj = datetime.strptime(value, fmt).date()
                                break
                            except ValueError:
                                continue
                                
                        if date_obj:
                            start_dt = datetime.combine(date_obj, time.min)
                            end_dt = datetime.combine(date_obj, time.max)
                            domain.append([base_field, '>=', start_dt.isoformat()])
                            domain.append([base_field, '<=', end_dt.isoformat()])
                            continue
                    except Exception as e:
                        _logger.error("Error parsing day interval: %s - %s", value, str(e))
                    
                # Parse the date and build a range domain if no specific handling above
                date_start, date_end = _parse_date_from_string(str(value), return_range=True)
                if date_start and date_end:
                    domain.append([base_field, '>=', date_start])
                    domain.append([base_field, '<=', date_end])
                    continue
                
                # Fallback to direct comparison if parsing failed
                domain.append([field, '=', value])
                continue
        
        # Handle many2one fields (stored as tuples or lists in read_group results)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            value = value[0]  # Use ID for domain
            
        # Add regular field condition
        if value is not None:
            # For fields with interval notation, use the base field name
            field_name = field.split(':')[0] if ':' in field else field
            domain.append([field_name, '=', value])
    
    # Return empty list if domain is identical to base_domain
    return domain if domain else []


def _parse_date_from_string(date_str, return_range=False):
    """
    Parse date from string in various formats, with special handling for week format.
    
    Args:
        date_str: String representation of date (e.g., 'W16 2025', '2025-04')
        return_range: If True, returns start and end date objects for the period
        
    Returns:
        Date object or tuple of (start_date, end_date) if return_range is True
    """
    # Handle week format (e.g., 'W16 2025')
    week_match = re.match(r'W(\d{1,2})\s+(\d{4})', date_str)
    if week_match:
        week_num = int(week_match.group(1))
        year = int(week_match.group(2))
        
        # Get the first day of the specified week
        first_day = datetime.strptime(f'{year}-{week_num}-1', '%Y-%W-%w')
        # Adjust to the beginning of the week
        start_date = first_day - timedelta(days=first_day.weekday())
        end_date = start_date + timedelta(days=6, hours=23, minutes=59, seconds=59)
        
        if return_range:
            return start_date.date(), end_date.date()
        return start_date.date()
    
    # Try various date formats
    formats = ['%Y-%m-%d', '%Y-%m', '%b %Y', '%Y%m%d']
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str, fmt)
            if return_range:
                if fmt == '%Y-%m':
                    last_day = calendar.monthrange(dt.year, dt.month)[1]
                    return dt.date(), date(dt.year, dt.month, last_day)
                else:
                    return dt.date(), dt.date()
            return dt.date()
        except ValueError:
            continue
    
    # Return as is if parsing fails
    return date_str


def _transform_graph_data(results, group_by_list, measures, base_domain, order_string=None):
    """
    Transform read_group results into the expected format for graph visualization.
    
    Args:
        results: Results from read_group query
        group_by_list: List of grouping fields with their configuration
        measures: List of measure fields with aggregation type
        base_domain: Original domain filter
        order_string: Optional order string (e.g. 'create_date asc')
        
    Returns:
        Transformed data for graph visualization
    """
    # Determine the primary grouping field (first in the list)
    primary_field = group_by_list[0].get('field') if group_by_list else None
    if not primary_field:
        return []
    
    # Get the interval if any
    primary_interval = group_by_list[0].get('interval')
    primary_field_with_interval = f"{primary_field}:{primary_interval}" if primary_interval else primary_field
    
    # Process secondary groupings (if any)
    secondary_fields = []
    for i, gb in enumerate(group_by_list[1:], 1):
        field = gb.get('field')
        interval = gb.get('interval')
        if field:
            field_with_interval = f"{field}:{interval}" if interval else field
            secondary_fields.append((field, field_with_interval))
    
    # Group by primary field first
    primary_groups = {}
    for result in results:
        # Extract the primary field value
        primary_value = None
        
        # Try format with interval first (standard for read_group)
        if primary_field_with_interval in result:
            primary_value = result[primary_field_with_interval]
        # Then try without interval
        elif primary_field in result:
            primary_value = result[primary_field]
        
        # Try with .get for default values if still None
        if primary_value is None:
            primary_value = result.get(primary_field_with_interval, result.get(primary_field))
        
        # Skip literal None values (but not date strings)
        if primary_value is None and not isinstance(primary_value, str):
            continue
        
        # Format primary value for cleaner display
        formatted_primary_value = primary_value
        
        # Create a hashable key for dictionary lookups
        dict_key = primary_value
        
        # For many2one fields as tuples (id, name)
        if isinstance(primary_value, tuple) and len(primary_value) == 2:
            formatted_primary_value = primary_value[1]
            dict_key = primary_value  # tuples are already hashable
        
        # For many2one fields from _get_field_values as dict {'id': id, 'display_name': name}
        elif isinstance(primary_value, dict) and 'display_name' in primary_value:
            formatted_primary_value = primary_value['display_name']
            # Convert dict to a hashable tuple
            dict_key = (primary_value.get('id'), primary_value.get('display_name'))
        
        # Handle date fields
        elif isinstance(primary_value, str):
            # Check if it's a date string format
            if primary_field_with_interval.endswith(':day') or \
               primary_field_with_interval.endswith(':week') or \
               primary_field_with_interval.endswith(':month') or \
               primary_field_with_interval.endswith(':quarter') or \
               primary_field_with_interval.endswith(':year'):
                formatted_primary_value = primary_value
                dict_key = primary_value
        
        # Create or get the group for this primary value
        if dict_key not in primary_groups:
            # Construct domain based on data type
            if primary_field_with_interval.endswith(':day') or \
               primary_field_with_interval.endswith(':week') or \
               primary_field_with_interval.endswith(':month') or \
               primary_field_with_interval.endswith(':quarter') or \
               primary_field_with_interval.endswith(':year'):
                base_field = primary_field_with_interval.split(':')[0]
                domain_field = base_field
            else:
                domain_field = primary_field
            
            primary_groups[dict_key] = {
                'key': str(formatted_primary_value),
                'odash.domain': _build_odash_domain({domain_field: primary_value})
            }
        
        # Process secondary fields and measures if they exist
        if secondary_fields:
            for sec_field, sec_field_with_interval in secondary_fields:
                sec_value = result.get(sec_field_with_interval)
                
                # Add measure values with secondary field in the key
                for measure in measures:
                    field = measure.get('field')
                    agg = measure.get('aggregation', 'sum')
                    
                    # Format the secondary field value
                    formatted_sec_value = sec_value
                    
                    # For many2one fields as tuples
                    if sec_value and isinstance(sec_value, tuple) and len(sec_value) == 2:
                        formatted_sec_value = sec_value[1]
                    
                    # For many2one fields as dict
                    elif sec_value and isinstance(sec_value, dict) and 'display_name' in sec_value:
                        formatted_sec_value = sec_value['display_name']
                    
                    # Construct the key for this measure and secondary field value
                    measure_key = f"{field}|{formatted_sec_value}" if sec_field else field
                    
                    # Get the measure value
                    if agg == 'count':
                        measure_value = result.get('__count', 0)
                    else:
                        measure_value = result.get(field, 0)
                    
                    # Add to the primary group
                    primary_groups[dict_key][measure_key] = measure_value
        # If no secondary fields, add measures directly to primary groups
        else:
            for measure in measures:
                field = measure.get('field')
                agg = measure.get('aggregation', 'sum')
                
                # Get the measure value
                if agg == 'count':
                    measure_value = result.get('__count', 0)
                else:
                    measure_value = result.get(field, 0)
                
                # Add to the primary group
                primary_groups[dict_key][field] = measure_value
    
    # Convert dictionary to list
    transformed_data = list(primary_groups.values())
    
    # Sort data if order string specified
    if order_string:
        # Extract field and direction
        parts = order_string.strip().split()
        sort_field = parts[0].strip() if len(parts) >= 1 else None
        sort_direction = parts[1].lower() if len(parts) >= 2 and parts[1].lower() in ['asc', 'desc'] else 'asc'
        
        if sort_field:
            # Sort by key for primary field
            if sort_field == primary_field:
                transformed_data.sort(key=lambda x: x.get('key', ''), reverse=(sort_direction == 'desc'))
            # Sort by measure value otherwise
            else:
                transformed_data.sort(key=lambda x: x.get(sort_field, 0), reverse=(sort_direction == 'desc'))
    
    return transformed_data


def process_dashboard_request(request_data, env):
    """
    Process dashboard visualization requests.
    This function handles validation, parsing, and routing to appropriate processor functions.
    
    Args:
        request_data: JSON data from the request, can be a single configuration or a list
        env: Odoo environment from the request
        
    Returns:
        Dictionary with results for each requested visualization
    """
    results = {}
    
    # Ensure request_data is a list
    if not isinstance(request_data, list):
        request_data = [request_data]
    
    # Process each visualization request
    for config in request_data:
        config_id = config.get('id')
        if not config_id:
            continue

        try:
            # Extract configuration parameters
            viz_type = config.get('type')
            model_name = config.get('model')
            data_source = config.get('data_source', {})

            # Validate essential parameters
            if not all([viz_type, model_name]):
                results[config_id] = {'error': 'Missing required parameters: type, model'}
                continue

            # Check if model exists
            try:
                model = env[model_name].sudo()
            except KeyError:
                results[config_id] = {'error': f'Model not found: {model_name}'}
                continue

            # Extract common parameters
            domain = data_source.get('domain', [])
            group_by = data_source.get('groupBy', [])
            order_by = data_source.get('orderBy', {})
            order_string = None
            if order_by:
                field = order_by.get('field')
                direction = order_by.get('direction', 'asc')
                if field:
                    order_string = f"{field} {direction}"

            # Check if SQL request is provided
            sql_request = data_source.get('sqlRequest')

            # Process based on visualization type
            if sql_request and viz_type in ['graph', 'table']:
                # Handle SQL request (with security measures)
                results[config_id] = _process_sql_request(sql_request, viz_type, config, env)
            elif viz_type == 'block':
                results[config_id] = _process_block(model, domain, config)
            elif viz_type == 'graph':
                results[config_id] = _process_graph(model, domain, group_by, order_string, config)
            elif viz_type == 'table':
                results[config_id] = _process_table(model, domain, group_by, order_string, config)
            else:
                results[config_id] = {'error': f'Unsupported visualization type: {viz_type}'}

        except Exception as e:
            _logger.exception("Error processing visualization %s:", config_id)
            results[config_id] = {'error': str(e)}
    
    return results


def _get_field_values(model, field_name, domain=None):
    """Get all possible values for a field in the model."""
    domain = domain or []
    field_info = model._fields.get(field_name)
    
    if not field_info:
        return []
    
    if field_info.type == 'selection':
        # Return all selection options
        return [key for key, _ in field_info.selection]
    
    elif field_info.type == 'many2one':
        # Return all possible values for the relation
        relation_model = model.env[field_info.comodel_name]
        rel_values = relation_model.search_read([], ['id', 'display_name'])
        return [{'id': r['id'], 'display_name': r['display_name']} for r in rel_values]
    
    # For other field types, retrieve actual values
    query = f"""
        SELECT DISTINCT {field_name} 
        FROM {model._table} 
        WHERE {field_name} IS NOT NULL
        LIMIT 1000
    """
    model.env.cr.execute(query)
    results = model.env.cr.fetchall()
    return [r[0] for r in results if r[0]]


def _build_date_range(model, field_name, domain, interval='month'):
    """Build a range of dates for show_empty functionality."""
    # Get min and max dates directly from database for better performance
    query = f"""
        SELECT MIN({field_name}), MAX({field_name})
        FROM {model._table}
        WHERE {field_name} IS NOT NULL
    """
    model.env.cr.execute(query)
    result = model.env.cr.fetchone()
    
    if not result or not result[0] or not result[1]:
        return []
    
    min_date = result[0]
    max_date = result[1]
    
    # Ensure dates are datetime objects
    if isinstance(min_date, str):
        min_date = datetime.strptime(min_date, '%Y-%m-%d')
    if isinstance(max_date, str):
        max_date = datetime.strptime(max_date, '%Y-%m-%d')
        
    # Add a buffer to ensure we include the full range
    if interval == 'day':
        current = min_date.replace(hour=0, minute=0, second=0, microsecond=0)
        result = []
        while current <= max_date:
            result.append(current.strftime('%Y-%m-%d'))
            current += timedelta(days=1)
        return result
    
    elif interval == 'week':
        # Start with the beginning of the week containing min_date
        current = min_date - timedelta(days=min_date.weekday())
        result = []
        while current <= max_date:
            week_num = current.strftime('%W')
            year = current.strftime('%Y')
            result.append(f"W{int(week_num)} {year}")
            current += timedelta(days=7)
        return result
    
    elif interval == 'month':
        current = min_date.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        result = []
        while current <= max_date:
            # Use full month name (e.g., "April 2025" instead of "Apr 2025")
            month_name = current.strftime('%B')
            year = current.strftime('%Y')
            result.append(f"{month_name} {year}")
            
            # Move to next month
            if current.month == 12:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=current.month + 1)
        return result
    
    elif interval == 'quarter':
        # Start with the beginning of the quarter containing min_date
        quarter = (min_date.month - 1) // 3 + 1
        current = min_date.replace(month=(quarter-1)*3+1, day=1)
        result = []
        while current <= max_date:
            quarter = (current.month - 1) // 3 + 1
            year = current.strftime('%Y')
            result.append(f"Q{quarter} {year}")
            
            # Move to next quarter
            if quarter == 4:
                current = current.replace(year=current.year + 1, month=1)
            else:
                current = current.replace(month=quarter*3+1)
        return result
    
    elif interval == 'year':
        current_year = min_date.year
        result = []
        while current_year <= max_date.year:
            result.append(str(current_year))
            current_year += 1
        return result
    
    return []


def _generate_empty_combinations(model, group_by_list, domain, results):
    """Generate all combinations for fields with show_empty=True.
    Takes into account existing values for fields without show_empty.
    """
    if not group_by_list or not results:
        return results
        
    # Initialize field values for each groupby field
    field_values = {}
    show_empty_fields = []
    
    for gb in group_by_list:
        field = gb.get('field')
        interval = gb.get('interval')
        show_empty = gb.get('show_empty', False)
        
        if not field:
            continue
            
        field_with_interval = f"{field}:{interval}" if interval else field
        
        if show_empty:
            show_empty_fields.append((field, interval, field_with_interval))
            
            # For date fields with interval, get range of possible values
            if interval and model._fields[field].type in ('date', 'datetime'):
                field_values[field_with_interval] = _build_date_range(model, field, domain, interval)
            else:
                # For non-date fields, get all possible values
                field_values[field_with_interval] = _get_field_values(model, field, domain)
        else:
            # For fields without show_empty, use only values present in results
            field_values[field_with_interval] = list(set(r[field_with_interval] for r in results if field_with_interval in r))
    
    # If no show_empty fields, return original results
    if not show_empty_fields:
        return results
    
    # Generate all possible combinations
    fields_to_combine = list(field_values.keys())
    values_to_combine = [field_values[f] for f in fields_to_combine]
    
    # Check if values are available for all fields
    if not all(values_to_combine):
        return results
        
    all_combinations = list(itertools.product(*values_to_combine))
    
    # Convert combinations to dictionaries
    combo_dicts = []
    for combo in all_combinations:
        combo_dict = {}
        for i, field in enumerate(fields_to_combine):
            combo_dict[field] = combo[i]
        combo_dicts.append(combo_dict)
    
    # Check which combinations already exist in results
    existing_combos = []
    for result in results:
        existing_combo = {}
        for field in fields_to_combine:
            if field in result:
                existing_combo[field] = result[field]
        existing_combos.append(existing_combo)
    
    # Find missing combinations
    missing_combos = []
    for combo in combo_dicts:
        if combo not in existing_combos:
            missing_combos.append(combo)
    
    # Add missing combinations to results
    for combo in missing_combos:
        new_result = combo.copy()
        
        # Add count as 0 for missing combinations
        new_result['__count'] = 0
        
        # Add default values for measure fields
        for result in results:
            for key, value in result.items():
                if key not in new_result and key != '__count' and not any(key.startswith(f) for f in fields_to_combine):
                    new_result[key] = 0
        
        results.append(new_result)
    
    return results


def _handle_show_empty(results, model, group_by_list, domain, measures=None):
    """Handle show_empty for groupBy fields by filling in missing combinations."""
    if not results or not group_by_list:
        return results
        
    # Check if any groupBy field has show_empty=True
    has_show_empty = any(gb.get('show_empty', False) for gb in group_by_list)
    if not has_show_empty:
        return results
        
    # Generate all combinations with show_empty fields
    return _generate_empty_combinations(model, group_by_list, domain, results)


def _process_block(model, domain, config):
    """Process block type visualization."""
    block_options = config.get('block_options', {})
    field = block_options.get('field')
    aggregation = block_options.get('aggregation', 'sum')
    label = block_options.get('label', field)
    
    if not field:
        return {'error': 'Missing field in block_options'}
    
    # Compute the aggregated value
    if aggregation == 'count':
        count = model.search_count(domain)
        return {
            'data': {
                'value': count,
                'label': label or 'Count',
                'odash.domain': []
            }
        }
    else:
        # For sum, avg, min, max
        try:
            # Use SQL for better performance on large datasets
            agg_func = aggregation.upper()
            
            # Build the WHERE clause and parameters securely
            if not domain:
                where_clause = "TRUE"
                where_params = []
            else:
                # Instead of using _where_calc directly, use search to get the query
                # This is a safer and more robust way to generate the WHERE clause
                records = model.search(domain)
                if not records:
                    where_clause = "FALSE"  # No matching records
                    where_params = []
                else:
                    id_list = records.ids
                    where_clause = f"{model._table}.id IN %s"
                    where_params = [tuple(id_list) if len(id_list) > 1 else (id_list[0],)]
                
            # More reliable and unified solution for all aggregations
            try:
                _logger.info("Processing %s aggregation for field %s", agg_func, field)
                
                # First check if there are any records
                count_query = f"""
                    SELECT COUNT(*) as count
                    FROM {model._table}
                    WHERE {where_clause}
                """
                model.env.cr.execute(count_query, where_params)
                count_result = model.env.cr.fetchone()
                count = 0
                if count_result and len(count_result) > 0:
                    count = count_result[0] if count_result[0] is not None else 0
                
                _logger.info("Found %s records matching the criteria", count)
                
                # If no records, return 0 for all aggregations
                if count == 0:
                    value = 0
                    _logger.info("No records found, using default value 0")
                else:
                    # Calculate the aggregation based on type
                    if agg_func == 'AVG':
                        # Calculate the sum for average
                        sum_query = f"""
                            SELECT SUM({field}) as total
                            FROM {model._table}
                            WHERE {where_clause}
                        """
                        model.env.cr.execute(sum_query, where_params)
                        sum_result = model.env.cr.fetchone()
                        total = 0
                        
                        if sum_result and len(sum_result) > 0:
                            total = sum_result[0] if sum_result[0] is not None else 0
                        
                        # Calculate the average
                        value = total / count if count > 0 else 0
                        _logger.info("Calculated AVG manually: total=%s, count=%s, avg=%s", total, count, value)
                    elif agg_func == 'MAX':
                        # Calculate the maximum
                        max_query = f"""
                            SELECT {field} as max_value
                            FROM {model._table}
                            WHERE {where_clause} AND {field} IS NOT NULL
                            ORDER BY {field} DESC
                            LIMIT 1
                        """
                        model.env.cr.execute(max_query, where_params)
                        max_result = model.env.cr.fetchone()
                        value = 0
                        
                        if max_result and len(max_result) > 0:
                            value = max_result[0] if max_result[0] is not None else 0
                        
                        _logger.info("Calculated MAX manually: %s", value)
                    elif agg_func == 'MIN':
                        # Calculate the minimum
                        min_query = f"""
                            SELECT {field} as min_value
                            FROM {model._table}
                            WHERE {where_clause} AND {field} IS NOT NULL
                            ORDER BY {field} ASC
                            LIMIT 1
                        """
                        model.env.cr.execute(min_query, where_params)
                        min_result = model.env.cr.fetchone()
                        value = 0
                        
                        if min_result and len(min_result) > 0:
                            value = min_result[0] if min_result[0] is not None else 0
                        
                        _logger.info("Calculated MIN manually: %s", value)
                    elif agg_func == 'SUM':
                        # Calculate the sum
                        sum_query = f"""
                            SELECT SUM({field}) as total
                            FROM {model._table}
                            WHERE {where_clause}
                        """
                        model.env.cr.execute(sum_query, where_params)
                        sum_result = model.env.cr.fetchone()
                        value = 0
                        
                        if sum_result and len(sum_result) > 0:
                            value = sum_result[0] if sum_result[0] is not None else 0
                        
                        _logger.info("Calculated SUM manually: %s", value)
                    else:
                        # Unrecognized aggregation function
                        value = 0
                        _logger.warning("Unrecognized aggregation function: %s", agg_func)
            except Exception as e:
                _logger.exception("Error calculating %s for %s: %s", agg_func, field, e)
                value = 0
            
            return {
                'data': {
                    'value': value,
                    'label': label or field,
                    'aggregation': aggregation,
                    'odash.domain': []
                }
            }
            
        except Exception as e:
            _logger.exception("Error in _process_block: %s", e)
            return {'error': f'Error processing block: {str(e)}'}


def _process_sql_request(sql_request, viz_type, config, env):
    """Process a SQL request with security measures."""
    # SECURITY WARNING: Direct SQL execution from API requests is risky.
    # This implementation includes safeguards but should be further reviewed.
    
    config_id = config.get('id')
    try:
        # Check for dangerous keywords (basic sanitization)
        dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'CREATE', 'ALTER', 'TRUNCATE']
        has_dangerous_keyword = any(keyword in sql_request.upper() for keyword in dangerous_keywords)
        
        if has_dangerous_keyword:
            _logger.warning("Dangerous SQL detected for config ID %s: %s", config_id, sql_request)
            return {'error': 'SQL contains prohibited operations'}
        
        # Execute the SQL query (with LIMIT safeguard)
        if 'LIMIT' not in sql_request.upper():
            sql_request += " LIMIT 1000"  # Default limit for safety
        
        try:
            env.cr.execute(sql_request)
            results = env.cr.dictfetchall()
            
            # Format data based on visualization type
            if viz_type == 'graph':
                return {'data': results}  # Simple pass-through for now
            elif viz_type == 'table':
                return {'data': results, 'metadata': {'total_count': len(results)}}
            
        except Exception as e:
            _logger.error("SQL execution error: %s", e)
            return {'error': f'SQL error: {str(e)}'}
            
    except Exception as e:
        _logger.exception("Error in _process_sql_request:")
        return {'error': str(e)}
    
    return {'error': 'Unexpected error in SQL processing'}


def _process_table(model, domain, group_by_list, order_string, config):
    """Process table type visualization."""
    table_options = config.get('table_options', {})
    columns = table_options.get('columns', [])
    limit = table_options.get('limit', 50)
    offset = table_options.get('offset', 0)
    
    if not columns:
        return {'error': 'Missing columns configuration for table'}
    
    # Extract fields to read
    fields_to_read = [col.get('field') for col in columns if col.get('field')]
    
    # Check if grouping is required
    if group_by_list:
        # Table with grouping - use read_group
        groupby_fields = []
        has_show_empty = any(gb.get('show_empty', False) for gb in group_by_list)
        
        for gb in group_by_list:
            field = gb.get('field')
            interval = gb.get('interval')
            if field:
                groupby_fields.append(f"{field}:{interval}" if interval else field)
                if field not in fields_to_read:
                    fields_to_read.append(field)
        
        if not groupby_fields:
            return {'error': "Invalid 'groupBy' configuration for grouped table"}
        
        # Add __count field for the counts per group
        fields_to_read.append('__count')
        
        try:
            # Execute read_group
            results = model.read_group(
                domain,
                fields=fields_to_read,
                groupby=groupby_fields,
                orderby=order_string,
                lazy=False
            )
            
            # Handle show_empty if needed
            if has_show_empty:
                results = _handle_show_empty(results, model, group_by_list, domain)
            
            # Format for table display
            total_count = len(results)
            results = results[offset:offset+limit] if limit else results
            
            # Add domain for each row - uniquement les critères de regroupement, sans le domaine d'entrée
            for result in results:
                row_domain = []  # Démarrer avec un domaine vide, sans inclure le domaine d'entrée
                
                # Add domain elements for each groupby field
                for gb_field in groupby_fields:
                    base_field = gb_field.split(':')[0] if ':' in gb_field else gb_field
                    value = result.get(gb_field)
                    
                    if value is not None:
                        if gb_field.endswith(':month') or gb_field.endswith(':week') or gb_field.endswith(':day') or gb_field.endswith(':year'):
                            # Handle date intervals
                            base_field = gb_field.split(':')[0]
                            interval = gb_field.split(':')[1]
                            
                            # Parse the date and build a range domain
                            date_start, date_end = _parse_date_from_string(str(value), return_range=True)
                            if date_start and date_end:
                                row_domain.append([base_field, '>=', date_start.isoformat()])
                                row_domain.append([base_field, '<=', date_end.isoformat()])
                        else:
                            # Direct comparison for regular fields
                            row_domain.append([base_field, '=', value])
                
                result['odash.domain'] = row_domain
            
            return {
                'data': results,
                'metadata': {
                    'page': offset // limit + 1 if limit else 1,
                    'limit': limit,
                    'total_count': total_count
                }
            }
            
        except Exception as e:
            _logger.exception("Error in _process_table with groupBy: %s", e)
            return {'error': f'Error processing grouped table: {str(e)}'}
    
    else:
        # Simple table - use search_read
        try:
            # Count total records for pagination
            total_count = model.search_count(domain)
            
            # Fetch the records
            records = model.search_read(
                domain,
                fields=fields_to_read,
                limit=limit,
                offset=offset,
                order=order_string
            )
            
            # Add domain for each record - uniquement l'ID, sans le domaine d'entrée
            for record in records:
                record['odash.domain'] = [('id', '=', record['id'])]
            
            return {
                'data': records,
                'metadata': {
                    'page': offset // limit + 1 if limit else 1,
                    'limit': limit,
                    'total_count': total_count
                }
            }
            
        except Exception as e:
            _logger.exception("Error in _process_table: %s", e)
            return {'error': f'Error processing table: {str(e)}'}


def _process_graph(model, domain, group_by_list, order_string, config):
    """Process graph type visualization."""
    graph_options = config.get('graph_options', {})
    measures = graph_options.get('measures', [])
    
    # Validate configuration
    if not group_by_list:
        return {'error': 'Missing groupBy configuration for graph'}
    
    # Default count measure if none provided
    if not measures:
        measures = [{'field': 'id', 'aggregation': 'count', 'label': 'Count'}]
    
    # Prepare groupby fields for read_group
    groupby_fields = []
    for gb in group_by_list:
        field = gb.get('field')
        interval = gb.get('interval')
        if field:
            groupby_fields.append(f"{field}:{interval}" if interval else field)
    
    # Prepare measure fields for read_group
    measure_fields = []
    for measure in measures:
        field = measure.get('field')
        agg = measure.get('aggregation', 'sum')
        if field and agg != 'count':
            measure_fields.append(field)
    
    # Execute read_group
    try:
        results = model.read_group(
            domain,
            fields=measure_fields,
            groupby=groupby_fields,
            orderby=order_string,
            lazy=False
        )
        
        # Handle show_empty if needed
        has_show_empty = any(gb.get('show_empty', False) for gb in group_by_list)
        if has_show_empty:
            results = _handle_show_empty(results, model, group_by_list, domain, measures)
        
        # Transform results into the expected format
        transformed_data = _transform_graph_data(results, group_by_list, measures, domain, order_string)
        
        return {'data': transformed_data}
        
    except Exception as e:
        _logger.exception("Error in _process_graph: %s", e)
        return {'error': f'Error processing graph data: {str(e)}'}


def _transform_graph_data(results, group_by_list, measures, base_domain, order_string=None):
    """
    Transform read_group results into the expected format for graph visualization.
    
    Args:
        results: Results from read_group query
        group_by_list: List of grouping fields with their configuration
        measures: List of measure fields with aggregation type
        base_domain: Original domain filter
        order_string: Optional order string (e.g. 'create_date asc' or 'amount_total desc')
    """
    # Determine the primary grouping field (first in the list)
    primary_field = group_by_list[0].get('field') if group_by_list else None
    if not primary_field:
        return []

    # Get the interval if any
    primary_interval = group_by_list[0].get('interval')
    primary_field_with_interval = f"{primary_field}:{primary_interval}" if primary_interval else primary_field

    # Process secondary groupings (if any)
    secondary_fields = []
    for i, gb in enumerate(group_by_list[1:], 1):
        field = gb.get('field')
        interval = gb.get('interval')
        if field:
            field_with_interval = f"{field}:{interval}" if interval else field
            secondary_fields.append((field, field_with_interval))

    # Initialize output data
    transformed_data = []

    # Group by primary field first
    primary_groups = {}
    for result in results:
        # Extract the primary field value - ATTENTION aux différents formats de clés
        primary_value = None

        # Essayer d'abord avec le format field:interval (standard de read_group)
        if primary_field_with_interval in result:
            primary_value = result[primary_field_with_interval]
        # Puis essayer avec le format field sans interval (utilisé par _handle_show_empty)
        elif primary_field in result:
            primary_value = result[primary_field]

        # Si on n'a toujours pas de valeur, essayer avec .get pour les valeurs par défaut
        if primary_value is None:
            primary_value = result.get(primary_field_with_interval, result.get(primary_field))

        # Filtrer uniquement les valeurs None littérales qui créent la clé "None"
        # mais pas les dates générées par _handle_show_empty
        if primary_value is None and not isinstance(primary_value, str):
            continue

        # Format primary value for cleaner display
        formatted_primary_value = primary_value

        # Create a hashable key for dictionary lookups
        dict_key = primary_value

        # For many2one fields as tuples (id, name)
        if isinstance(primary_value, tuple) and len(primary_value) == 2:
            formatted_primary_value = primary_value[1]
            dict_key = primary_value  # tuples are already hashable

        # For many2one fields from _get_field_values as dict {'id': id, 'display_name': name}
        elif isinstance(primary_value, dict) and 'display_name' in primary_value:
            formatted_primary_value = primary_value['display_name']
            # Convert dict to a hashable tuple (id, name) for use as a key
            dict_key = (primary_value.get('id'), primary_value.get('display_name'))

        # Handle date fields (crucial for show_empty)
        elif isinstance(primary_value, str):
            # Check if it's a date string format
            if primary_field_with_interval.endswith(':day') or \
                    primary_field_with_interval.endswith(':week') or \
                    primary_field_with_interval.endswith(':month') or \
                    primary_field_with_interval.endswith(':quarter') or \
                    primary_field_with_interval.endswith(':year'):
                formatted_primary_value = primary_value
                dict_key = primary_value

        # Create or get the group for this primary value
        if dict_key not in primary_groups:
            # Construire le domaine en fonction du type de donnée
            if primary_field_with_interval.endswith(':day') or \
                    primary_field_with_interval.endswith(':week') or \
                    primary_field_with_interval.endswith(':month') or \
                    primary_field_with_interval.endswith(':quarter') or \
                    primary_field_with_interval.endswith(':year'):
                base_field = primary_field_with_interval.split(':')[0]
                domain_field = base_field
            else:
                domain_field = primary_field

            primary_groups[dict_key] = {
                'key': str(formatted_primary_value),
                'odash.domain': _build_odash_domain({domain_field: primary_value})
            }

        # Process secondary fields and measures if available
        if secondary_fields:
            # Process secondary fields and measures
            for sec_field, sec_field_with_interval in secondary_fields:
                sec_value = result.get(sec_field_with_interval)

                # Add measure values with secondary field in the key
                for measure in measures:
                    field = measure.get('field')
                    agg = measure.get('aggregation', 'sum')

                    # Format the secondary field value correctly
                    formatted_sec_value = sec_value

                    # For many2one fields as tuples (id, name)
                    if isinstance(sec_value, tuple) and len(sec_value) == 2:
                        formatted_sec_value = sec_value[1]

                    # Use string for secondary key
                    sec_key = str(formatted_sec_value) if formatted_sec_value is not None else 'None'

                    # Calculate the combined key for this measure - use field|sec_key format
                    # This maintains the format utilisé dans l'API original
                    measure_key = f"{field}|{sec_key}"

                    # Add the measure value to the primary group
                    field_agg_key = f"{field}:{agg}"
                    primary_groups[dict_key][measure_key] = result.get(field_agg_key, 0)
        else:
            # If no secondary grouping, add measures directly to primary group
            for measure in measures:
                field = measure.get('field')
                agg = measure.get('aggregation', 'sum')
                
                # Use just the field name as key, without the aggregation
                measure_key = field
                
                # Add the measure value directly to primary group
                field_agg_key = f"{field}:{agg}"
                primary_groups[dict_key][measure_key] = result.get(field_agg_key, 0)

    # Convert to list for output
    transformed_data = list(primary_groups.values())

    # If order_string is specified, try to sort the results
    if order_string and transformed_data:
        try:
            field, direction = order_string.split(' ')
            reverse = (direction.lower() == 'desc')
            transformed_data.sort(key=lambda x: x.get(field, 0), reverse=reverse)
        except Exception as e:
            _logger.warning("Error sorting graph data: %s", str(e))

    return transformed_data
