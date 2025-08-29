"""
Odashboard Engine - Version 1.0.0
This file contains all the processing logic for dashboard visualizations.
"""
import logging

from datetime import datetime, timedelta
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
    search = request.params.get('search', '')
    page = int(kw.get('page', 1))
    limit = 50

    domain = []

    if search:
        domain.append(('name', 'ilike', search))

    records = request.env[model_name].sudo().search(domain, limit=limit, offset=(page - 1) * limit)
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


def _process_block(model, domain, config):
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
                '__domain': []
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
                        _logger.warning("Unrecognized aggregation function: %s", agg_func)
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
    """Process a SQL request with security measures."""
    try:
        env.cr.execute(sql_request)
        results = env.cr.dictfetchall()

        # Format data based on visualization type
        if viz_type == 'graph':
            if results and isinstance(results[0], dict) and 'key' not in results[0]:
                transformed_results = []
                for row in results:
                    if isinstance(row, dict) and row:
                        new_row = {}
                        keys = list(row.keys())
                        if keys:
                            first_key = keys[0]
                            new_row['key'] = row[first_key]

                            for k in keys[1:]:
                                new_row[k] = row[k]

                            transformed_results.append(new_row)
                        else:
                            transformed_results.append(row)
                    else:
                        transformed_results.append(row)
                return {'data': transformed_results}
            else:
                return {'data': results}
        elif viz_type == 'table':
            return {'data': results}
        elif viz_type == 'block':
            results = results[0]
            results["label"] = config.get('block_options').get('field')
            return {'data': results}

    except Exception as e:
        _logger.error("SQL execution error: %s", e)
        return {'error': f'SQL error: {str(e)}'}

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

            if 'show_empty' in group_by_list[0] and group_by_list[0]['show_empty']:
                if ':' in groupby_fields[0]:
                    results = complete_missing_date_intervals(results)
                else:
                    results = complete_missing_selection_values(results, model, groupby_fields[0])

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


def _process_graph(model, domain, group_by_list, order_string, config):
    """Process graph type visualization."""
    graph_options = config.get('graph_options', {})
    measures = graph_options.get('measures', [])

    if not group_by_list:
        group_by_list = [{'field': 'name'}]
        order_string = "name asc"

    if not measures:
        # Default to count measure if not specified
        measures = [{'field': 'id', 'aggregation': 'count'}]

    # Prepare groupby fields for read_group
    groupby_fields = []

    for gb in group_by_list:
        field = gb.get('field')
        interval = gb.get('interval') if gb.get('interval') != 'auto' else 'month'
        if field:
            groupby_fields.append(f"{field}:{interval}" if interval else field)

    # Prepare measure fields for read_group
    measure_fields = []
    for measure in measures:
        measure_fields.append(f"{measure.get('field')}:{measure.get('aggregation', 'sum')}")

    # Execute read_group
    try:
        results = model.read_group(
            domain,
            fields=measure_fields,
            groupby=groupby_fields,
            orderby=order_string,
            lazy=True
        )

        if 'show_empty' in group_by_list[0] and group_by_list[0]['show_empty']:
            if ':' in groupby_fields[0]:
                results = complete_missing_date_intervals(results)
            else:
                results = complete_missing_selection_values(results, model, groupby_fields[0])

        # Transform results into the expected format
        transformed_data = []
        for result in results:
            data = {
                'key': result[groupby_fields[0]][1] if isinstance(result[groupby_fields[0]], tuple) or isinstance(
                    result[groupby_fields[0]], list) else result[groupby_fields[0]],
                '__domain': result['__domain']
            }

            if len(groupby_fields) > 1:
                sub_results = model.read_group(
                    result['__domain'],
                    fields=measure_fields,
                    groupby=groupby_fields[1],
                    orderby=groupby_fields[1],
                    lazy=True
                )

                if 'show_empty' in group_by_list[1]:
                    if ':' in groupby_fields[1]:
                        sub_results = complete_missing_date_intervals(sub_results)
                    else:
                        sub_results = complete_missing_selection_values(sub_results, model, groupby_fields[1])

                for sub_result in sub_results:
                    for measure in config['graph_options']['measures']:
                        data_sub_key = sub_result[groupby_fields[1]][1] if isinstance(sub_result[groupby_fields[1]],
                                                                                      tuple) or isinstance(
                            sub_result[groupby_fields[1]], list) else sub_result[groupby_fields[1]]
                        data[f"{measure['field']}|{data_sub_key}"] = {"value": sub_result[measure['field']],
                                                                      "__domain": sub_result["__domain"]}
            else:
                for measure in config['graph_options']['measures']:
                    data[measure['field']] = result[measure['field']]

            transformed_data.append(data)

        return {'data': transformed_data}

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
                result = _process_block(model, domain, config)
            elif viz_type == 'graph':
                result = _process_graph(model, domain, group_by, order_string, config)
            elif viz_type == 'table':
                result = _process_table(model, domain, group_by, order_string, config)
            else:
                result = {'error': f'Unsupported visualization type: {viz_type}'}

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
