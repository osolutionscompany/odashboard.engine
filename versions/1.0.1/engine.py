import logging

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import pytz

_logger = logging.getLogger(__name__)


def _has_company_field(model):
    """
    Check if a model has company_id or company_ids field
    
    Args:
        model: Odoo model object
        
    Returns:
        str or False: 'company_id', 'company_ids', or False if no company field
    """
    if 'company_id' in model._fields:
        return 'company_id'
    elif 'company_ids' in model._fields:
        return 'company_ids'
    return False


def _apply_company_filtering(domain, model, env):
    """
    Apply company filtering to a domain if the model has company_id or company_ids field

    Args:
        domain: Existing domain filter
        model: Odoo model object
        env: Odoo environment

    Returns:
        list: Domain with company filtering applied
    """
    company_field = _has_company_field(model)
    if not company_field:
        _logger.debug("Model %s has no company field, skipping company filtering", model._name)
        return domain

    try:
        dashboard = env.context.get('dashboard_id')
        if not dashboard or not dashboard.allowed_company_ids:
            _logger.debug("No company filtering applied - no dashboard or allowed companies")
            return domain

        company_ids = dashboard.allowed_company_ids.ids
        _logger.info("Applying company filter for companies: %s on field: %s", company_ids, company_field)

        # Create appropriate domain based on field type
        company_domain = []
        if company_field == 'company_id':
            # Many2one field: records with no company OR records with allowed company
            company_domain = ['|', ('company_id', '=', False), ('company_id', 'in', company_ids)]
        elif company_field == 'company_ids':
            # Many2many field: records with no companies OR records with any allowed company
            company_domain = ['|', ('company_ids', '=', False), ('company_ids', 'in', company_ids)]

        # Combine existing domain with company domain
        if domain:
            return domain + company_domain
        else:
            return company_domain

    except Exception as e:
        _logger.warning("Error applying company filtering: %s", e)
        return domain


def _enrich_group_by_with_labels(group_by_list, model):
    """Enrich groupBy objects with field labels for frontend display."""
    if not group_by_list:
        return group_by_list

    enriched_group_by = []
    for gb in group_by_list:
        enriched_gb = gb.copy()
        field_name = gb.get('field')
        if field_name:
            field_info = model._fields.get(field_name)
            if field_info:
                enriched_gb['label'] = field_info.string or field_name.replace('_', ' ').title()
            else:
                enriched_gb['label'] = field_name.replace('_', ' ').title()
        enriched_group_by.append(enriched_gb)

    return enriched_group_by


def _format_datetime_value(value, field_type, lang=None, user_timezone=None):
    """
    Format date/datetime values with locale and Odoo user timezone support for data tables

    Args:
        value: The datetime/date value from database
        field_type: 'date' or 'datetime'
        lang: Odoo res.lang record
        user_timezone: User timezone from env.user.tz (e.g., 'Europe/Paris', 'America/New_York')

    Returns:
        Formatted string optimized for table display with locale and timezone support
    """
    if not value:
        return value

    try:
        # Parse the datetime value
        if isinstance(value, str):
            dt = datetime.fromisoformat(value.replace('T', ' ').replace('Z', ''))
        elif hasattr(value, 'strftime'):
            dt = value
        else:
            return str(value)

        # Convert to user's timezone if provided and it's a datetime field
        if field_type == 'datetime' and user_timezone:
            try:
                # Assume database datetime is in UTC if no timezone info
                if dt.tzinfo is None:
                    dt = pytz.UTC.localize(dt)

                # Convert to user's timezone
                user_tz = pytz.timezone(user_timezone)
                dt = dt.astimezone(user_tz)
            except Exception as e:
                _logger.warning("Error converting timezone for %s to %s: %s", dt, user_timezone, e)
                # Continue with original datetime if timezone conversion fails

        # Use Odoo language record formatting if available
        if lang and hasattr(lang, 'date_format'):
            try:
                if field_type == 'datetime':
                    # Combine date_format with short_time_format for datetime
                    date_fmt = lang.date_format or '%m/%d/%Y'
                    time_fmt = lang.short_time_format or '%H:%M'
                    combined_fmt = f"{date_fmt} {time_fmt}"
                    return dt.strftime(combined_fmt)
                else:
                    # Use only date_format for date fields
                    date_fmt = lang.date_format or '%m/%d/%Y'
                    return dt.strftime(date_fmt)
            except Exception as e:
                _logger.warning("Error using Odoo language format %s: %s", lang.date_format if lang else 'None', e)
                # Fall through to default formatting

        # Fallback to default formatting if no language record or formatting fails
        if field_type == 'datetime':
            return dt.strftime('%d/%m/%Y %H:%M')  # Default European format
        else:
            return dt.strftime('%d/%m/%Y')

    except Exception as e:
        _logger.warning("Error formatting datetime value %s: %s", value, e)
        return str(value)


def get_user_context(env):
    """
    Get current user's context information for cache invalidation

    Args:
        env: Odoo environment from the request

    Returns:
        Dictionary with user language, timezone, and date format settings
    """
    try:
        user = env.user

        # Get user language record
        user_lang_code = user.lang if hasattr(user, 'lang') else 'en_US'
        lang_record = env['res.lang']._lang_get(user_lang_code) if user_lang_code else None

        # Get user timezone
        user_timezone = user.tz if hasattr(user, 'tz') else 'UTC'

        # Prepare user context data
        context_data = {
            'lang': user_lang_code,
            'tz': user_timezone,
            'date_format': lang_record.date_format if lang_record and hasattr(lang_record,
                                                                              'date_format') else '%m/%d/%Y',
            'time_format': lang_record.short_time_format if lang_record and hasattr(lang_record,
                                                                                    'short_time_format') else '%H:%M'
        }

        return {'success': True, 'data': context_data}

    except Exception as e:
        _logger.error("Error in get_user_context: %s", str(e))
        return {'success': False, 'error': str(e)}


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
                              'web.', 'auth.', 'wizard.']

        for prefix in technical_prefixes:
            domain.append(('model', 'not like', f'{prefix}%'))

        # Models starting with underscore
        domain.append(('model', 'not like', '\\_%'))

        # Execute the optimized search
        model_obj = env['ir.model'].sudo()
        models = model_obj.search(domain)

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

    :param model_name: Name of the Odoo model (example: 'sale.order')
    :return: JSON with information about the model's fields
    """
    try:
        _logger.info("API call: Fetching fields info for model: %s", model_name)

        # Check if the model exists
        if model_name not in env:
            return {'success': False, 'error': f"Model '{model_name}' not found"}

        # Get field information
        model_obj = env[model_name].sudo()
        fields_info = _get_fields_info(model_obj)

        return {'success': True, 'data': fields_info}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_model_records(model_name, kw, env):
    """
    Retrieve all records of a specific model with pagination and search functionality.

    :param model_name: Name of the Odoo model (example: 'res.partner')
    :param page: Page number for pagination (default: 1)
    :param search: Optional search string to filter records by name (default: '')
    :return: JSON with the model records
    """
    try:
        _logger.info("API call: Fetching records for model: %s", model_name)

        # Check if the model exists
        if model_name not in env:
            return {'success': False, 'error': f"Model '{model_name}' not found"}

        # Get pagination parameters
        page = int(kw.get('page', 1))
        limit = 50  # Number of records per page
        offset = (page - 1) * limit

        # Get search parameter
        search_query = kw.get('search', '')

        # Create domain for search
        domain = []
        if search_query:
            domain.append(('name', 'ilike', search_query))

        # Get model
        model = env[model_name].sudo()

        # Apply company filtering
        domain = _apply_company_filtering(domain, model, env)

        # Count total records matching the domain
        total_records = model.search_count(domain)
        total_pages = (total_records + limit - 1) // limit

        # Search with pagination
        records = model.search(domain, order="name asc", limit=limit, offset=offset)

        # Format the records
        record_list = []
        for record in records:
            record_data = {
                'id': record.id,
                'name': record.name,
            }

            # Include display_name if different from name
            if record.display_name != record.name:
                record_data['display_name'] = record.display_name

            # Get other basic fields if they exist
            for field in ['active', 'code', 'ref']:
                if hasattr(record, field):
                    record_data[field] = getattr(record, field)

            record_list.append(record_data)

        return {'success': True, 'data': record_list}

    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_model_search(model_name, kw, request):
    search = kw.get('search', '')
    page = int(kw.get('page', 1))
    limit = 50

    domain = []

    if search:
        domain.append(('name', 'ilike', search))

    # Get model and apply company filtering
    model = request.env[model_name].sudo()
    domain = _apply_company_filtering(domain, model, request.env)

    records = model.search(domain, limit=limit, offset=(page - 1) * limit)
    record_list = []
    for record in records:
        record_list.append({
            'id': record.id,
            'name': record.name,
        })

    return {'success': True, 'data': record_list}


def _get_fields_info(model):
    """
    Get information about all fields of an Odoo model.

    :param model: Odoo model object
    :return: List of field information
    """
    fields_info = []

    # Get fields from the model
    fields_data = model.fields_get()

    for field_name, field_data in fields_data.items():
        field_type = field_data.get('type', 'unknown')

        # Check if it's a computed field that's not stored
        field_obj = model._fields.get(field_name)
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

        if field_obj and field_obj.comodel_name:
            field_info['model'] = field_obj.comodel_name

        # Add selection options if field is a selection
        if field_data.get('type') == 'selection' and 'selection' in field_data:
            field_info['selection'] = [
                {'value': value, 'label': label}
                for value, label in field_data['selection']
            ]

        fields_info.append(field_info)

    # Sort fields by name for better readability
    fields_info.sort(key=lambda x: x['name'])

    return fields_info


def _process_block(model, domain, config, env=None):
    block_options = config.get('block_options', {})
    field = block_options.get('field')
    aggregation = block_options.get('aggregation', 'sum')
    label = block_options.get('label', field)

    if not field:
        return {'error': 'Missing field in block_options'}

    # Apply company filtering if env is provided
    if env:
        domain = _apply_company_filtering(domain, model, env)

    # Count total records for metadata
    total_count = model.search_count(domain)

    # Compute the aggregated value
    if aggregation == 'count':
        return {
            'data': {
                'value': total_count,
                'label': label or 'Count',
                '__domain': []
            },
            'metadata': {
                'total_count': total_count
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
                    # Calculate aggregation based on type
                    if agg_func == 'AVG':
                        # Calculate sum for average
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

                        # Calculate average
                        value = total / count if count > 0 else 0
                        _logger.info("Calculated AVG manually: total=%s, count=%s, avg=%s", total, count, value)
                    elif agg_func == 'MAX':
                        # Calculate maximum
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
                        # Calculate minimum
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
                        # Calculate sum
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
            except Exception as e:
                _logger.exception("Error calculating %s for %s: %s", agg_func, field, e)
                value = 0

            return {
                'data': {
                    'value': value,
                    'label': label or f'{aggregation.capitalize()} of {field}',
                    '__domain': []
                }
            }
        except Exception as e:
            _logger.error("Error calculating block value: %s", e)
            return {'error': f'Error calculating {aggregation} for {field}: {str(e)}'}


def _process_sql_request(sql_request, viz_type, config, env):
    """
    Process a SQL request with security measures using odoo.tools.SQL.
    
    SECURITY: This function does NOT accept raw SQL from the client.
    Instead, it accepts structured parameters and builds SQL server-side.
    
    Expected sql_request structure:
    {
        "type": "custom_query",  # Type of query
        "model": "res.partner",  # Base model (for table name)
        "select": [              # Fields to select
            {"field": "name", "aggregation": "count"},
            {"field": "country_id", "as": "country"}
        ],
        "where": [               # WHERE conditions (safe parameters)
            {"field": "active", "operator": "=", "value": True},
            {"field": "customer_rank", "operator": ">", "value": 0}
        ],
        "group_by": ["country_id"],  # GROUP BY fields
        "order_by": [{"field": "name", "direction": "ASC"}],  # ORDER BY
        "limit": 100
    }
    
    Args:
        sql_request: Structured query parameters (NOT raw SQL)
        viz_type: Type of visualization (graph, table)
        config: Visualization configuration
        env: Odoo environment
        
    Returns:
        dict: Query results or error
    """
    try:
        from odoo.tools import SQL
        
        # Validate that sql_request is a dict with structured parameters
        if not isinstance(sql_request, dict):
            return {'error': 'SQL request must be a structured dictionary, not raw SQL'}
        
        # Check if this is a raw SQL string (security check)
        if isinstance(sql_request, str) or 'SELECT' in str(sql_request).upper():
            _logger.error("SECURITY: Attempted to pass raw SQL. Only structured queries are allowed.")
            return {'error': 'Raw SQL is not allowed. Use structured query parameters.'}
        
        # Extract query parameters
        query_type = sql_request.get('type', 'custom_query')
        model_name = sql_request.get('model')
        select_fields = sql_request.get('select', [])
        where_conditions = sql_request.get('where', [])
        group_by_fields = sql_request.get('group_by', [])
        order_by_fields = sql_request.get('order_by', [])
        limit = sql_request.get('limit', 1000)
        
        # Validate required parameters
        if not model_name:
            return {'error': 'Model name is required for SQL queries'}
        
        # Get the model and validate access
        try:
            model = env[model_name].sudo()
        except KeyError:
            return {'error': f'Model not found: {model_name}'}
        
        # Get table name from model
        table_name = model._table
        
        # Build SELECT clause
        select_parts = []
        for field_spec in select_fields:
            if isinstance(field_spec, str):
                # Simple field name
                field_name = field_spec
                select_parts.append(field_name)
            elif isinstance(field_spec, dict):
                field_name = field_spec.get('field')
                aggregation = field_spec.get('aggregation')
                alias = field_spec.get('as', field_name)
                
                if aggregation:
                    # Aggregation function
                    agg_upper = aggregation.upper()
                    if agg_upper in ['COUNT', 'SUM', 'AVG', 'MIN', 'MAX']:
                        select_parts.append(f"{agg_upper}({field_name}) as {alias}")
                    else:
                        return {'error': f'Unsupported aggregation: {aggregation}'}
                else:
                    select_parts.append(f"{field_name} as {alias}" if alias != field_name else field_name)
        
        if not select_parts:
            select_parts = ['*']
        
        select_clause = ', '.join(select_parts)
        
        # Build WHERE clause using parameterized queries
        where_parts = []
        where_params = []
        for condition in where_conditions:
            field = condition.get('field')
            operator = condition.get('operator', '=')
            value = condition.get('value')
            
            # Validate operator to prevent SQL injection
            allowed_operators = ['=', '!=', '>', '<', '>=', '<=', 'LIKE', 'ILIKE', 'IN', 'NOT IN', 'IS NULL', 'IS NOT NULL']
            if operator.upper() not in allowed_operators:
                return {'error': f'Unsupported operator: {operator}'}
            
            if operator.upper() in ['IS NULL', 'IS NOT NULL']:
                where_parts.append(f"{field} {operator.upper()}")
            elif operator.upper() in ['IN', 'NOT IN']:
                if not isinstance(value, (list, tuple)):
                    return {'error': f'Operator {operator} requires a list value'}
                placeholders = ','.join(['%s'] * len(value))
                where_parts.append(f"{field} {operator.upper()} ({placeholders})")
                where_params.extend(value)
            else:
                where_parts.append(f"{field} {operator} %s")
                where_params.append(value)
        
        where_clause = ' AND '.join(where_parts) if where_parts else '1=1'
        
        # Build GROUP BY clause
        group_by_clause = ''
        if group_by_fields:
            group_by_clause = 'GROUP BY ' + ', '.join(group_by_fields)
        
        # Build ORDER BY clause
        order_by_clause = ''
        if order_by_fields:
            order_parts = []
            for order_spec in order_by_fields:
                if isinstance(order_spec, str):
                    order_parts.append(order_spec)
                elif isinstance(order_spec, dict):
                    field = order_spec.get('field')
                    direction = order_spec.get('direction', 'ASC').upper()
                    if direction not in ['ASC', 'DESC']:
                        direction = 'ASC'
                    order_parts.append(f"{field} {direction}")
            order_by_clause = 'ORDER BY ' + ', '.join(order_parts)
        
        # Build LIMIT clause
        limit_clause = f'LIMIT {int(limit)}' if limit else ''
        
        # Construct the final SQL query using SQL wrapper
        query_parts = [
            f"SELECT {select_clause}",
            f"FROM {table_name}",
            f"WHERE {where_clause}",
            group_by_clause,
            order_by_clause,
            limit_clause
        ]
        
        # Remove empty parts
        query_sql = ' '.join([part for part in query_parts if part])
        
        _logger.info(f"Executing secure SQL query: {query_sql}")
        _logger.debug(f"Query parameters: {where_params}")
        
        # Execute the query with parameters
        env.cr.execute(query_sql, where_params)
        results = env.cr.dictfetchall()
        
        # Format results based on visualization type
        if viz_type == 'table':
            return {
                'data': results,
                'metadata': {
                    'total_count': len(results),
                    'limit': limit
                }
            }
        elif viz_type == 'graph':
            return {
                'data': results
            }
        else:
            return {'data': results}
            
    except Exception as e:
        _logger.exception("Error in _process_sql_request: %s", e)
        return {'error': f'Error processing SQL request: {str(e)}'}


def _process_table(model, domain, group_by_list, order_string, config, env=None):
    """Process table type visualization."""
    table_options = config.get('table_options', {})
    columns = table_options.get('columns', [])
    limit = table_options.get('limit', 50)
    offset = table_options.get('offset', 0)

    if not columns:
        return {'error': 'Missing columns configuration for table'}

    # Apply company filtering if env is provided
    if env:
        domain = _apply_company_filtering(domain, model, env)

    # Extract fields to read
    fields_to_read = [col.get('field') for col in columns if col.get('field')]

    # Simple table - use search_read
    try:
        # Count total records for pagination
        total_count = model.search_count(domain)

        if group_by_list:
            table_options = config.get('table_options', {})
            measures = table_options.get('columns', [])

            if not measures:
                # Default to count measure if not specified
                measures = [{'field': 'id', 'aggregation': 'count'}]

            measure_fields = []
            for measure in measures:
                measure_fields.append(f"{measure.get('field')}:{measure.get('aggregation', 'sum')}")

            # Prepare groupby fields for read_group
            groupby_fields = []

            for gb in group_by_list:
                field = gb.get('field')
                interval = gb.get('interval') if gb.get('interval') != 'auto' else 'month'
                if field:
                    groupby_fields.append(f"{field}:{interval}" if interval else field)

            results = model.read_group(
                domain,
                fields=measure_fields,
                groupby=groupby_fields,
                orderby=order_string,
                lazy=False
            )

            # Check if we should show empty values for the first group by
            show_empty = group_by_list[0].get('show_empty', False) if group_by_list else False

            if show_empty:
                if ':' in groupby_fields[0]:
                    results = complete_missing_date_intervals(results)
                else:
                    results = complete_missing_selection_values(results, model, groupby_fields[0])
            else:
                # Filter out empty values when show_empty is False
                results = [result for result in results if any(
                    isinstance(v, (int, float)) and v > 0
                    for k, v in result.items()
                    if k not in ['__domain', '__range'] and not k.startswith('__')
                )]

            transformed_data = []
            for result in results:
                data = {
                    'key': result[groupby_fields[0]][1] if isinstance(result[groupby_fields[0]],
                                                                      tuple) or isinstance(
                        result[groupby_fields[0]], list) else result[groupby_fields[0]],
                    '__domain': result['__domain']
                }

                for measure in measures:
                    data[measure['field']] = result[measure['field']]

                transformed_data.append(data)
        else:
            transformed_data = model.search_read(
                domain,
                fields=fields_to_read,
                limit=limit,
                offset=offset,
                order=order_string
            )

            for data in transformed_data:
                data['__domain'] = []
                for key in data.keys():
                    if isinstance(data[key], tuple):
                        data[key] = data[key][1]
                    # Format date/datetime fields for display
                    elif key in fields_to_read:
                        field_info = model._fields.get(key)
                        if field_info and field_info.type in ['date', 'datetime'] and data[key]:
                            # Get user timezone from Odoo user profile
                            user_timezone = model.env.user.tz if hasattr(model.env.user, 'tz') else None
                            # Detect user locale for proper date formatting
                            user_lang = model.env.user.lang if hasattr(model.env.user, 'lang') else None
                            lang = model.env['res.lang']._lang_get(user_lang)
                            data[key] = _format_datetime_value(data[key], field_info.type, lang, user_timezone)

        return {
            'data': transformed_data,
            'metadata': {
                'page': offset // limit + 1 if limit else 1,
                'limit': limit,
                'total_count': total_count
            }
        }

    except Exception as e:
        _logger.exception("Error in _process_table: %s", e)
        return {'error': f'Error processing table: {str(e)}'}


def _prepare_groupby_fields(group_by_list):
    """Prepare groupby fields for read_group operation."""
    groupby_fields = []
    for gb in group_by_list:
        field = gb.get('field')
        interval = gb.get('interval') if gb.get('interval') != 'auto' else 'month'
        if field:
            groupby_fields.append(f"{field}:{interval}" if interval else field)
    return groupby_fields


def _prepare_measures(measures, model):
    """
    Prepare measures for read_group, separating regular fields from relational fields.
    
    Supports advanced aggregations (SUM/AVG/MIN/MAX) on relational fields when 'related_field' is specified.
    
    Returns:
        tuple: (measure_fields, relational_measures) where:
            - measure_fields: List of fields ready for read_group
            - relational_measures: List of One2many/Many2many measures for special handling
    """
    measure_fields = []
    relational_measures = []
    
    for measure in measures:
        field_name = measure.get('field')
        aggregation = measure.get('aggregation', 'sum')
        related_field = measure.get('related_field')  # New parameter for advanced aggregations
        
        # Check if field exists and can be used in read_group
        if field_name in model._fields:
            field_info = model._fields[field_name]
            field_type = field_info.type
            
            # Handle relational fields specially
            if field_type in ['one2many', 'many2many']:
                # Validate aggregation type for relational fields
                if aggregation in ['count', 'count_distinct']:
                    # Count aggregations don't need related_field
                    relational_measures.append({
                        **measure,
                        'field_type': field_type,
                        'field_info': field_info
                    })
                    _logger.info(f"{field_type.title()} field '{field_name}' will be handled with special {aggregation} logic")
                    continue
                    
                elif aggregation in ['sum', 'avg', 'min', 'max']:
                    # Advanced aggregations require related_field
                    if not related_field:
                        _logger.warning(f"Skipping field '{field_name}' - {aggregation} aggregation on {field_type} requires 'related_field' parameter")
                        continue
                    
                    # Validate that related_field exists on the related model
                    try:
                        related_model = model.env[field_info.comodel_name]
                        if related_field not in related_model._fields:
                            _logger.warning(f"Skipping field '{field_name}' - related_field '{related_field}' not found on model '{field_info.comodel_name}'")
                            continue
                        
                        # Validate that related_field is numeric for sum/avg
                        related_field_info = related_model._fields[related_field]
                        if aggregation in ['sum', 'avg'] and related_field_info.type not in ['integer', 'float', 'monetary']:
                            _logger.warning(f"Skipping field '{field_name}' - {aggregation} aggregation requires numeric related_field, got '{related_field_info.type}'")
                            continue
                        
                        relational_measures.append({
                            **measure,
                            'field_type': field_type,
                            'field_info': field_info,
                            'related_field': related_field,
                            'related_field_info': related_field_info
                        })
                        _logger.info(f"{field_type.title()} field '{field_name}' will be handled with {aggregation}({related_field}) logic")
                        continue
                        
                    except Exception as e:
                        _logger.warning(f"Error validating related_field for '{field_name}': {e}")
                        continue
                else:
                    _logger.warning(f"Skipping field '{field_name}' - unsupported aggregation '{aggregation}' for {field_type} fields")
                    continue
        
        measure_fields.append(f"{field_name}:{aggregation}")
    
    return measure_fields, relational_measures


def _build_relational_metadata(relational_measures, model):
    """Build metadata for relational fields to optimize query generation."""
    field_metadata = {}
    
    for measure in relational_measures:
        field_name = measure.get('field')
        field_info = measure['field_info']
        field_type = measure['field_type']
        aggregation = measure.get('aggregation', 'count')
        related_field = measure.get('related_field')
        
        if field_type == 'one2many':
            related_model_name = field_info.comodel_name
            foreign_key = field_info.inverse_name
            related_model = model.env[related_model_name]
            
            field_metadata[field_name] = {
                'field_type': 'one2many',
                'related_model': related_model,
                'foreign_key': foreign_key,
                'table_name': related_model._table,
                'aggregation': aggregation,
                'related_field': related_field,
                'related_field_info': measure.get('related_field_info')
            }
            
        elif field_type == 'many2many':
            related_model_name = field_info.comodel_name
            relation_table = field_info.relation
            column1 = field_info.column1  # Foreign key to current model
            column2 = field_info.column2  # Foreign key to related model
            related_model = model.env[related_model_name]
            
            field_metadata[field_name] = {
                'field_type': 'many2many',
                'related_model': related_model,
                'relation_table': relation_table,
                'column1': column1,
                'column2': column2,
                'related_table_name': related_model._table,
                'aggregation': aggregation,
                'related_field': related_field,
                'related_field_info': measure.get('related_field_info')
            }
    
    return field_metadata


def _build_relational_query(metadata, record_ids_tuple):
    """
    Build optimized SQL query for relational field aggregations.
    
    Supports: COUNT, COUNT_DISTINCT, SUM, AVG, MIN, MAX on relational fields.
    For advanced aggregations, uses related_field parameter.
    """
    aggregation = metadata.get('aggregation', 'count')
    related_field = metadata.get('related_field')
    
    if metadata['field_type'] == 'one2many':
        table = metadata['table_name']
        where_clause = f"{metadata['foreign_key']} IN %s"
        
        # Build SELECT clause based on aggregation type
        if aggregation == 'count':
            select_clause = "COUNT(*)"
        elif aggregation == 'count_distinct':
            select_clause = "COUNT(DISTINCT id)"
        elif aggregation in ['sum', 'avg', 'min', 'max'] and related_field:
            # Advanced aggregations on related field
            agg_func = aggregation.upper()
            select_clause = f"{agg_func}({related_field})"
        else:
            # Fallback to count
            select_clause = "COUNT(*)"
            
        return f"SELECT {select_clause} FROM {table} WHERE {where_clause}"
            
    else:  # many2many
        relation_table = metadata['relation_table']
        related_table = metadata['related_table_name']
        column1 = metadata['column1']  # FK to source model
        column2 = metadata['column2']  # FK to target model
        
        if aggregation == 'count':
            # Simple count on relation table
            select_clause = "COUNT(*)"
            return f"SELECT {select_clause} FROM {relation_table} WHERE {column1} IN %s"
            
        elif aggregation == 'count_distinct':
            # Count distinct related records
            select_clause = f"COUNT(DISTINCT {column2})"
            return f"SELECT {select_clause} FROM {relation_table} WHERE {column1} IN %s"
            
        elif aggregation in ['sum', 'avg', 'min', 'max'] and related_field:
            # Advanced aggregations require JOIN with related table
            agg_func = aggregation.upper()
            select_clause = f"{agg_func}(rt.{related_field})"
            
            return f"""
                SELECT {select_clause} 
                FROM {relation_table} rel 
                JOIN {related_table} rt ON rel.{column2} = rt.id 
                WHERE rel.{column1} IN %s
            """
        else:
            # Fallback to count
            select_clause = "COUNT(*)"
            return f"SELECT {select_clause} FROM {relation_table} WHERE {column1} IN %s"


def _execute_relational_query(query, record_ids_tuple, model, field_name, metadata=None):
    """Execute relational query with fallback to ORM if SQL fails."""
    try:
        model.env.cr.execute(query, (record_ids_tuple,))
        result = model.env.cr.fetchone()[0]
        return result if result is not None else 0
    except Exception as e:
        _logger.warning(f"SQL query failed for {field_name}, falling back to ORM: {e}")
        
        # Enhanced ORM fallback that handles advanced aggregations
        if metadata:
            return _execute_orm_fallback(record_ids_tuple, model, field_name, metadata)
        else:
            # Simple count fallback for backward compatibility
            group_records = model.browse(list(record_ids_tuple))
            field_data = group_records.read([field_name])
            return sum(len(data.get(field_name, [])) for data in field_data)


def _execute_orm_fallback(record_ids_tuple, model, field_name, metadata):
    """
    Execute ORM fallback for advanced relational aggregations.
    
    This handles cases where SQL queries fail by using Odoo's ORM
    to compute the same aggregations.
    """
    aggregation = metadata.get('aggregation', 'count')
    related_field = metadata.get('related_field')
    
    try:
        group_records = model.browse(list(record_ids_tuple))
        
        if aggregation == 'count':
            # Simple count of related records
            field_data = group_records.read([field_name])
            return sum(len(data.get(field_name, [])) for data in field_data)
            
        elif aggregation == 'count_distinct':
            # Count distinct related records
            all_related_ids = set()
            for record in group_records:
                related_records = getattr(record, field_name, [])
                all_related_ids.update(related_records.ids)
            return len(all_related_ids)
            
        elif aggregation in ['sum', 'avg', 'min', 'max'] and related_field:
            # Advanced aggregations on related field values
            all_values = []
            
            for record in group_records:
                related_records = getattr(record, field_name, [])
                for related_record in related_records:
                    value = getattr(related_record, related_field, None)
                    if value is not None:
                        all_values.append(value)
            
            if not all_values:
                return 0
            
            if aggregation == 'sum':
                return sum(all_values)
            elif aggregation == 'avg':
                return sum(all_values) / len(all_values)
            elif aggregation == 'min':
                return min(all_values)
            elif aggregation == 'max':
                return max(all_values)
        
        # Fallback to count if aggregation not recognized
        field_data = group_records.read([field_name])
        return sum(len(data.get(field_name, [])) for data in field_data)
        
    except Exception as e:
        _logger.error(f"ORM fallback failed for {field_name}: {e}")
        return 0


def _compute_relational_aggregates(results, relational_measures, model):
    """
    Compute aggregates for One2many and Many2many fields using optimized SQL queries.
    
    Args:
        results: read_group results to enhance
        relational_measures: List of relational measures to compute
        model: Odoo model instance
    
    Returns:
        Enhanced results with relational aggregates
    """
    if not relational_measures:
        return results
    
    # Build metadata for all relational measures
    field_metadata = _build_relational_metadata(relational_measures, model)
    
    # Process each result group
    for result in results:
        # Get record IDs in this group
        group_domain = result.get('__domain', [])
        group_record_ids = model.search(group_domain, order='id').ids
        
        if not group_record_ids:
            # No records in this group, set all counts to 0
            for measure in relational_measures:
                result[measure.get('field')] = 0
            continue
        
        record_ids_tuple = tuple(group_record_ids)
        
        # Group measures by table/relation for batch processing
        measures_by_table = {}
        for measure in relational_measures:
            field_name = measure.get('field')
            metadata = field_metadata[field_name]
            
            # Use table_name for one2many, relation_table for many2many
            table_key = metadata.get('table_name') or metadata.get('relation_table')
            
            if table_key not in measures_by_table:
                measures_by_table[table_key] = []
            measures_by_table[table_key].append((field_name, metadata))
        
        # Execute optimized queries per table
        for table_key, table_measures in measures_by_table.items():
            for field_name, metadata in table_measures:
                query = _build_relational_query(metadata, record_ids_tuple)
                total_count = _execute_relational_query(query, record_ids_tuple, model, field_name, metadata)
                result[field_name] = total_count
    
    return results


def _apply_show_empty(results, group_by_list, groupby_fields, model):
    """Apply show_empty logic to results."""
    if not group_by_list or not groupby_fields:
        return results
    
    show_empty = group_by_list[0].get('show_empty', False)
    
    if show_empty:
        if ':' in groupby_fields[0]:
            results = complete_missing_date_intervals(results)
        else:
            results = complete_missing_selection_values(results, model, groupby_fields[0])
    else:
        # Filter out empty values when show_empty is False
        results = [result for result in results if any(
            isinstance(v, (int, float)) and v > 0
            for k, v in result.items()
            if k not in ['__domain', '__range'] and not k.startswith('__')
        )]
    
    return results


def _transform_results(results, groupby_fields, config, model):
    """Transform read_group results into the expected format."""
    transformed_data = []
    
    for result in results:
        data = {
            'key': result[groupby_fields[0]][1] if isinstance(result[groupby_fields[0]], (tuple, list)) else result[groupby_fields[0]],
            '__domain': result['__domain']
        }
        
        if len(groupby_fields) > 1:
            # Handle multi-level groupby recursively
            measure_fields = [f"{measure['field']}:{measure.get('aggregation', 'sum')}" 
                            for measure in config['graph_options']['measures']]
            
            sub_results = model.read_group(
                result['__domain'],
                fields=measure_fields,
                groupby=groupby_fields[1],
                orderby=groupby_fields[1],
                lazy=True
            )
            
            # Check if we should show empty values for the second group by
            show_empty_2 = config.get('group_by_list', [{}])[1].get('show_empty', False) if len(config.get('group_by_list', [])) > 1 else False
            
            if show_empty_2:
                if ':' in groupby_fields[1]:
                    sub_results = complete_missing_date_intervals(sub_results)
                else:
                    sub_results = complete_missing_selection_values(sub_results, model, groupby_fields[1])
            
            for sub_result in sub_results:
                for measure in config['graph_options']['measures']:
                    data_sub_key = sub_result[groupby_fields[1]][1] if isinstance(sub_result[groupby_fields[1]], (tuple, list)) else sub_result[groupby_fields[1]]
                    data[f"{measure['field']}|{data_sub_key}"] = {
                        "value": sub_result[measure['field']],
                        "__domain": sub_result["__domain"]
                    }
        else:
            for measure in config['graph_options']['measures']:
                data[measure['field']] = result[measure['field']]
        
        transformed_data.append(data)
    
    return transformed_data


def _process_graph(model, domain, group_by_list, order_string, config, env=None):
    """
    Process graph type visualization with optimized relational field handling.
    
    This function has been refactored into modular components for better maintainability,
    performance, and extensibility. It supports both One2many and Many2many field counting
    with advanced SQL optimizations.
    """
    try:
        # Apply company filtering if env is provided
        if env:
            domain = _apply_company_filtering(domain, model, env)
        
        # Count total records for metadata
        total_count = model.search_count(domain)
        
        # Set defaults if not provided
        if not group_by_list:
            group_by_list = [{'field': 'name'}]
            order_string = "name asc"
        
        graph_options = config.get('graph_options', {})
        measures = graph_options.get('measures', [])
        if not measures:
            measures = [{'field': 'id', 'aggregation': 'count'}]
        
        # Prepare groupby fields and measures
        groupby_fields = _prepare_groupby_fields(group_by_list)
        measure_fields, relational_measures = _prepare_measures(measures, model)
        
        # If no valid measure fields remain, use default count
        if not measure_fields and not relational_measures:
            measure_fields = ['id:count']
        
        # Execute standard read_group for regular fields
        results = model.read_group(
            domain,
            fields=measure_fields,
            groupby=groupby_fields,
            orderby=order_string,
            lazy=True
        )
        
        # Compute relational aggregates (One2many/Many2many) with optimized SQL
        results = _compute_relational_aggregates(results, relational_measures, model)
        
        # Apply show_empty logic
        results = _apply_show_empty(results, group_by_list, groupby_fields, model)
        
        # Transform results into expected format
        # Pass group_by_list in config for multi-level groupby support
        config_with_groupby = {**config, 'group_by_list': group_by_list}
        transformed_data = _transform_results(results, groupby_fields, config_with_groupby, model)
        
        return {
            'data': transformed_data,
            'metadata': {
                'total_count': total_count
            }
        }
        
    except Exception as e:
        _logger.exception("Error in _process_graph: %s", e)
        return {'error': f'Error processing graph data: {str(e)}'}


def complete_missing_selection_values(results, model, field_name):
    """
    Fills in missing values in the results for fields of type selection or many2one

    Args:
     results (list): The read_group results
     model (Model): The Odoo model on which the read_group was performed
     field_name (str): The name of the field (without grouping suffix)

    Returns:
     list: List completed with missing values
    """
    if not results:
        return results

    field_info = model._fields.get(field_name)
    if not field_info:
        return results

    field_type = field_info.type
    if field_type not in ['selection', 'many2one']:
        return results

    all_possible_values = []

    if field_type == 'selection':
        if callable(field_info.selection):
            selection_options = field_info.selection(model)
        else:
            selection_options = field_info.selection
        all_possible_values = [value for value, _ in selection_options]

    elif field_type == 'many2one':
        related_model = model.env[field_info.comodel_name].sudo()
        all_possible_values = related_model.search([]).ids

    present_values = set()
    groupby_field = field_name

    for result in results:
        for key in result.keys():
            if key.split(':')[0] == field_name:
                groupby_field = key
                break

    for result in results:
        if groupby_field in result and result[groupby_field] is not None:
            value = result[groupby_field]
            if isinstance(value, (list, tuple)) and len(value) >= 2:
                present_values.add(value[0])
            else:
                present_values.add(value)

    missing_values = [v for v in all_possible_values if v not in present_values]

    complete_results = list(results)

    template = results[0] if results else None

    if template and missing_values:
        for missing_value in missing_values:
            new_entry = {k: 0 if isinstance(v, (int, float)) else (None if v is None else v)
                         for k, v in template.items() if k != groupby_field}

            if field_type == 'selection':
                domain = [(field_name, '=', missing_value)]
            else:  # many2one
                domain = [(field_name, '=', missing_value)]

            new_entry[groupby_field] = missing_value

            if field_type == 'many2one' and missing_value:
                related_model = model.env[field_info.comodel_name].sudo()
                record = related_model.browse(missing_value)
                if record.exists():
                    new_entry[groupby_field] = [missing_value, record.display_name]

            if '__domain' in template:
                new_entry['__domain'] = domain

            if '__context' in template:
                new_entry['__context'] = template['__context']

            complete_results.append(new_entry)

    return complete_results


def complete_missing_date_intervals(results):
    """
    Fills in the missing intervals in the read_group results

    Args:
     results (list): Read_group results containing __range

    Returns:
     list: List completed with missing intervals
    """
    if not results or len(results) < 2:
        return results

    complete_results = [results[0]]  # Start with the first result

    interval_type = None
    range_field = None

    for key in results[0]['__range']:
        if key.endswith(':day'):
            interval_type = 'day'
            range_field = key
            break
        elif key.endswith(':week'):
            interval_type = 'week'
            range_field = key
            break
        elif key.endswith(':month'):
            interval_type = 'month'
            range_field = key
            break
        elif key.endswith(':quarter'):
            interval_type = 'quarter'
            range_field = key
            break
        elif key.endswith(':year'):
            interval_type = 'year'
            range_field = key
            break

    if not interval_type:
        return results

    for i in range(1, len(results)):
        prev_result = complete_results[-1]
        curr_result = results[i]

        try:
            prev_to = datetime.strptime(prev_result['__range'][range_field]['to'], '%Y-%m-%d %H:%M:%S')
            curr_from = datetime.strptime(curr_result['__range'][range_field]['from'], '%Y-%m-%d %H:%M:%S')
        except Exception:
            prev_to = datetime.strptime(prev_result['__range'][range_field]['to'], '%Y-%m-%d')
            curr_from = datetime.strptime(curr_result['__range'][range_field]['from'], '%Y-%m-%d')

        if prev_to < curr_from:
            next_date = prev_to

            while next_date < curr_from:
                if interval_type == 'day':
                    interval_end = next_date + timedelta(days=1)
                    label = next_date.strftime("%d %b %Y")
                elif interval_type == 'week':
                    interval_end = next_date + timedelta(weeks=1)
                    label = f"W{interval_end.isocalendar()[1]} {interval_end.year}"
                elif interval_type == 'month':
                    interval_end = next_date + relativedelta(months=1)
                    label = next_date.strftime('%B %Y')
                elif interval_type == 'quarter':
                    interval_end = next_date + relativedelta(months=3)
                    quarter = (next_date.month - 1) // 3 + 1
                    label = f"Q{quarter} {next_date.year}"
                elif interval_type == 'year':
                    interval_end = next_date + relativedelta(years=1)
                    label = str(next_date.year)

                base_field = range_field.split(':')[0]
                domain = [
                    '&',
                    (base_field, '>=', next_date.strftime('%Y-%m-%d %H:%M:%S')),
                    (base_field, '<', interval_end.strftime('%Y-%m-%d %H:%M:%S'))
                ]

                missing_result = {
                    range_field: label,
                    '__range': {
                        range_field: {
                            'from': next_date.strftime('%Y-%m-%d %H:%M:%S'),
                            'to': interval_end.strftime('%Y-%m-%d %H:%M:%S')
                        }
                    },
                    '__domain': domain,
                    '__context': curr_result.get('__context', {})
                }

                for key, value in curr_result.items():
                    if key not in [range_field, '__range', '__domain', '__context']:
                        if isinstance(value, (int, float)):
                            missing_result[key] = 0
                        elif value is None:
                            missing_result[key] = None
                        else:
                            missing_result[key] = value

                complete_results.append(missing_result)
                next_date = interval_end

        complete_results.append(curr_result)

    return complete_results


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

            # Enrich groupBy with field labels for frontend display
            group_by = _enrich_group_by_with_labels(group_by, model)

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
                result = _process_sql_request(sql_request, viz_type, config, env)
            elif viz_type == 'block':
                result = _process_block(model, domain, config, env)
            elif viz_type == 'graph':
                result = _process_graph(model, domain, group_by, order_string, config, env)
            elif viz_type == 'table':
                result = _process_table(model, domain, group_by, order_string, config, env)
            else:
                result = {'error': f'Unsupported visualization type: {viz_type}'}

            # Add enriched groupBy to result for frontend access
            if group_by and viz_type in ['graph', 'table']:
                result['enriched_group_by'] = group_by

            if data_source.get('preview') and viz_type != 'block':
                result['data'] = result['data'][:50]

            results[config_id] = result


        except Exception as e:
            _logger.exception("Error processing visualization %s:", config_id)
            results[config_id] = {'error': str(e)}

    return results


def get_action_config(action_name):
    """
    Define action configurations for the unified API system.
    This allows the engine to define its own action mappings without requiring
    updates to the customer-installed odashboard module.

    Args:
        action_name (str): The action to get configuration for

    Returns:
        dict: Configuration with success/error format
    """
    try:
        # Define all available actions and their configurations
        action_configs = {
            'get_user_context': {
                'method': 'get_user_context',
                'args': ['env'],
                'required_params': [],
                'description': 'Get current user context (language, timezone, date formats) for cache invalidation'
            },
            'get_models': {
                'method': 'get_models',
                'args': ['env'],
                'required_params': [],
                'description': 'Get list of models relevant for analytics'
            },
            'get_model_fields': {
                'method': 'get_model_fields',
                'args': [{'param': 'model_name'}, 'env'],
                'required_params': ['model_name'],
                'description': 'Get fields information for a specific model'
            },
            'get_model_records': {
                'method': 'get_model_records',
                'args': [{'param': 'model_name'}, 'parameters', 'env'],
                'required_params': ['model_name'],
                'description': 'Get records of a specific model with pagination'
            },
            'get_model_search': {
                'method': 'get_model_search',
                'args': [{'param': 'model_name'}, 'parameters', 'request'],
                'required_params': ['model_name'],
                'description': 'Search records of a specific model'
            },
            'process_dashboard_request': {
                'method': 'process_dashboard_request',
                'args': [{'param': 'request_data', 'default': 'parameters'}, 'env'],
                'required_params': ['request_data'],
                'description': 'Process dashboard visualization requests'
            }
        }

        if action_name in action_configs:
            return {'success': True, 'data': action_configs[action_name]}
        else:
            return {
                'success': False,
                'error': f'Unknown action: {action_name}. Available actions: {", ".join(action_configs.keys())}'
            }

    except Exception as e:
        _logger.error("Error in get_action_config: %s", str(e))
        return {'success': False, 'error': str(e)}
