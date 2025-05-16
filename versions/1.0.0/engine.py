"""
Odashboard Engine - Version 1.0.0
This file contains all the processing logic for dashboard visualizations.
"""
import json
import logging
import itertools
from datetime import datetime, date, timedelta
import calendar
import re
from dateutil.relativedelta import relativedelta

_logger = logging.getLogger(__name__)


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


def _parse_date_from_string(date_str, return_range=False):
    """Parse a date string in various formats and return a datetime object.
    If return_range is True, return a tuple of start and end dates for period formats.
    """
    if not date_str:
        return None
    
    # Week pattern (e.g., W16 2025)
    week_pattern = re.compile(r'W(\d{1,2})\s+(\d{4})')
    week_match = week_pattern.match(date_str)
    if week_match:
        week_num = int(week_match.group(1))
        year = int(week_match.group(2))
        # Get the first day of the week
        first_day = datetime.strptime(f'{year}-{week_num}-1', '%Y-%W-%w').date()
        if return_range:
            last_day = first_day + timedelta(days=6)
            return first_day, last_day
        return first_day
    
    # Month pattern (e.g., January 2025 or 2025-01)
    month_pattern = re.compile(r'(\w+)\s+(\d{4})|(\d{4})-(\d{2})')
    month_match = month_pattern.match(date_str)
    if month_match:
        if month_match.group(1) and month_match.group(2):
            # Format: January 2025
            month_name = month_match.group(1)
            year = int(month_match.group(2))
            month_num = datetime.strptime(month_name, '%B').month
        else:
            # Format: 2025-01
            year = int(month_match.group(3))
            month_num = int(month_match.group(4))
        
        if return_range:
            first_day = date(year, month_num, 1)
            last_day = date(year, month_num, calendar.monthrange(year, month_num)[1])
            return first_day, last_day
        return date(year, month_num, 1)
    
    # Standard date format
    try:
        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').date()
        if return_range:
            return parsed_date, parsed_date
        return parsed_date
    except ValueError:
        pass
    
    # ISO format
    try:
        parsed_date = datetime.fromisoformat(date_str).date()
        if return_range:
            return parsed_date, parsed_date
        return parsed_date
    except ValueError:
        pass
    
    return None


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


def _build_odash_domain(group_by_values):
    """Build odash.domain for a specific data point based on groupby values.
    Returns only the specific domain for this data point, not including the base domain.
    """
    domain = []
    
    for field, value in group_by_values.items():
        # Skip count field
        if field == '__count':
            continue
            
        # Handle interval notation in field name (e.g., create_date:month)
        if ':' in field:
            base_field, interval = field.split(':')
            
            # Handle date intervals
            if interval in ('day', 'week', 'month', 'quarter', 'year'):
                # Parse the date and build a range domain
                date_start, date_end = _parse_date_from_string(str(value), return_range=True)
                if date_start and date_end:
                    domain.append([base_field, '>=', date_start])
                    domain.append([base_field, '<=', date_end])
                continue
        
        # Handle many2one fields (stored as tuples or lists in read_group results)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            value = value[0]  # Use ID for domain
            
        # Add regular field condition
        if value is not None:
            domain.append([field.split(':')[0], '=', value])
    
    return domain


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
    """Transform read_group results into the expected format for graph visualization.
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
                if sec_value and isinstance(sec_value, tuple) and len(sec_value) == 2:
                    formatted_sec_value = sec_value[1]

                # For many2one fields from _get_field_values as dict {'id': id, 'display_name': name}
                elif sec_value and isinstance(sec_value, dict) and 'display_name' in sec_value:
                    formatted_sec_value = sec_value['display_name']  # display name for cleaner output

                # Construct the key for this measure and secondary field value
                measure_key = f"{field}|{formatted_sec_value}" if sec_field else field

                # Get the measure value from the result
                if agg == 'count':
                    measure_value = result.get('__count', 0)
                else:
                    measure_value = result.get(field, 0)

                # Add to the primary group
                primary_groups[dict_key][measure_key] = measure_value

    # Convert the dictionary to a list
    transformed_data = list(primary_groups.values())

    # Trier les données selon le champ de tri spécifié
    # Analyser order_string pour détecter la direction de tri
    sort_direction = 'asc'  # Par défaut
    sort_field = None

    if order_string:
        # Extraire le champ et la direction du order_string
        parts = order_string.strip().split()
        if len(parts) >= 1:
            sort_field = parts[0].strip()
        if len(parts) >= 2 and parts[1].lower() in ['asc', 'desc']:
            sort_direction = parts[1].lower()

    # Si pas de champ de tri spécifié, utiliser le premier groupby
    if not sort_field and group_by_list:
        primary_gb = group_by_list[0]
        sort_field = primary_gb.get('field')

    if sort_field:
        try:
            # Log pour débogage
            _logger.info("Sorting by field %s with direction %s", sort_field, sort_direction)

            # Pour les dates avec formatage "DD MMM YYYY", convertir en dates pour tri correct
            if sort_field in ['date', 'create_date', 'write_date'] or sort_field.endswith('_date'):
                # Fonction pour extraire la date d'une clé au format texte
                def extract_date(item):
                    # Gérer le cas où item est une chaîne directement
                    if isinstance(item, str):
                        key = item
                    else:
                        # Sinon c'est un dictionnaire avec une clé 'key'
                        key = item.get('key', '')

                    try:
                        # Traitement spécial pour les mois au format "Apr 2025" ou "April 2025"
                        if ' ' in key and not key.startswith('W') and not key.startswith('Q'):
                            try:
                                month_part, year_part = key.split(' ')
                                # Table de correspondance pour les noms de mois complets et abréviations
                                month_map = {
                                    'Jan': 1, 'January': 1,
                                    'Feb': 2, 'February': 2,
                                    'Mar': 3, 'March': 3,
                                    'Apr': 4, 'April': 4,
                                    'May': 5, 'May': 5,
                                    'Jun': 6, 'June': 6,
                                    'Jul': 7, 'July': 7,
                                    'Aug': 8, 'August': 8,
                                    'Sep': 9, 'Sept': 9, 'September': 9,
                                    'Oct': 10, 'October': 10,
                                    'Nov': 11, 'November': 11,
                                    'Dec': 12, 'December': 12
                                }

                                if month_part in month_map:
                                    month_num = month_map[month_part]
                                    year_num = int(year_part)
                                    # Créer la date du premier jour du mois
                                    date_obj = datetime(year_num, month_num, 1)
                                    _logger.info("Converted month %s to date %s", key, date_obj)
                                    return date_obj
                            except Exception as e:
                                _logger.error("Failed to parse month format %s: %s", key, e)

                        # Traitement spécial pour les semaines au format "W15 2025"
                        if key.startswith('W') and ' ' in key:
                            try:
                                week_part, year_part = key.split(' ')
                                week_num = int(week_part[1:])  # Enlever le 'W' et convertir en nombre
                                year_num = int(year_part)

                                # Créer une date pour le premier jour de l'année
                                first_day = datetime(year_num, 1, 1)

                                # Ajouter le nombre de semaines (chaque semaine = 7 jours)
                                # On soustrait 1 car W1 correspond à la première semaine
                                date_obj = first_day + timedelta(days=(week_num - 1) * 7)
                                return date_obj
                            except Exception as e:
                                _logger.error("Failed to parse week format %s: %s", key, e)

                        # Essayer divers formats de date standards
                        formats = ['%d %b %Y', '%Y-%m-%d', '%Y-%m', '%m %Y']
                        for fmt in formats:
                            try:
                                date_obj = datetime.strptime(key, fmt)
                                return date_obj
                            except ValueError:
                                continue
                        # Si aucun format ne correspond, utiliser la clé telle quelle
                        return key
                    except Exception as e:
                        _logger.error("Error parsing date %s: %s", key, e)
                        return key

                # Trier par date, en respectant la direction
                reverse = (sort_direction == 'desc')
                # Log avant tri
                _logger.info("Before sorting: %s", [item.get('key') for item in transformed_data])

                # Débugging des dates
                for item in transformed_data:
                    if isinstance(item, dict):
                        key = item.get('key', '')
                    else:
                        key = str(item)
                    date_value = extract_date(item)
                    _logger.info("Key: %s => Date value: %s", key, date_value)

                transformed_data.sort(key=extract_date, reverse=reverse)
                # Log après tri
                _logger.info("After sorting (reverse=%s): %s", reverse, [item.get('key') for item in transformed_data])
            else:
                # Tri normal par clé, en respectant la direction
                reverse = (sort_direction == 'desc')
                transformed_data.sort(key=lambda x: x.get('key', ''), reverse=reverse)
        except Exception as e:
            _logger.warning("Error sorting graph data: %s", e)

    return transformed_data
