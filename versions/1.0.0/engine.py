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
    """Build odash.domain for a specific data point based on groupby values.
    Returns only the specific domain for this data point, not including the base domain.
    """
    domain = []
    
    for field, value in group_by_values.items():
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
        
        # Handle week pattern specifically
        if isinstance(value, str) and re.match(r'W\d{1,2}\s+\d{4}', value):
            # Handle week format by getting date range
            start_date, end_date = _parse_date_from_string(value, return_range=True)
            domain.append([field, '>=', start_date.isoformat()])
            domain.append([field, '<=', end_date.isoformat()])
        elif field.endswith(':month') or field.endswith(':week') or field.endswith(':day') or field.endswith(':year'):
            # Handle date intervals
            base_field = field.split(':')[0]
            interval = field.split(':')[1]
            
            if interval == 'month' and re.match(r'\d{4}-\d{2}', str(value)):
                year, month = str(value).split('-')
                start_date = date(int(year), int(month), 1)
                end_date = date(int(year), int(month), calendar.monthrange(int(year), int(month))[1])
                domain.append([base_field, '>=', start_date.isoformat()])
                domain.append([base_field, '<=', end_date.isoformat()])
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
                
                # Default fallback if parsing fails
                domain.append([field, '=', value])
            else:
                # Direct comparison for other formats
                domain.append([field, '=', value])
        else:
            # Regular field
            domain.append([field, '=', value])
        
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
        
        # Process secondary fields and measures if they exist
        if secondary_fields:
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
                        formatted_sec_value = sec_value['display_name'] # display name for cleaner output
                    
                    # Construct the key for this measure and secondary field value
                    measure_key = f"{field}|{formatted_sec_value}" if sec_field else field
                    
                    # Get the measure value from the result
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
                
                # Get the measure value from the result
                if agg == 'count':
                    measure_value = result.get('__count', 0)
                else:
                    measure_value = result.get(field, 0)
                
                # Add to the primary group
                primary_groups[dict_key][field] = measure_value
    
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
                        # Traitement spécial pour les dates au format "DD MMM YYYY" (ex: "11 Apr 2025")
                        if ' ' in key and not key.startswith('W') and not key.startswith('Q'):
                            try:
                                parts = key.split(' ')
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
                                
                                # Format "DD MMM YYYY" (ex: "11 Apr 2025")
                                if len(parts) == 3 and parts[1] in month_map:
                                    day_num = int(parts[0])
                                    month_num = month_map[parts[1]]
                                    year_num = int(parts[2])
                                    date_obj = datetime(year_num, month_num, day_num)
                                    _logger.info("Key: %s => Date value: %s", key, date_obj)
                                    return date_obj
                                # Format "MMM YYYY" (ex: "Apr 2025")
                                elif len(parts) == 2 and parts[0] in month_map:
                                    month_num = month_map[parts[0]]
                                    year_num = int(parts[1])
                                    # Créer la date du premier jour du mois
                                    date_obj = datetime(year_num, month_num, 1)
                                    _logger.info("Key: %s => Date value: %s", key, date_obj)
                                    return date_obj
                            except Exception as e:
                                _logger.error("Failed to parse date format %s: %s", key, e)
                        
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
                                date_obj = first_day + timedelta(days=(week_num-1)*7)
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
    
    elif field_info.type in ['date', 'datetime']:
        # Cette partie est gérée séparément avec _build_date_range
        # basé sur l'intervalle et le domaine
        return []
    
    else:
        # Pour les autres types de champs, récupérer toutes les valeurs existantes
        records = model.search(domain)
        # Filtrer les valeurs None pour éviter les problèmes
        values = [v for v in list(set(records.mapped(field_name))) if v is not None]
        return values


def _build_date_range(model, field_name, domain, interval='month'):
    """Build a range of dates for show_empty functionality."""
    # Approche plus simple et robuste - ignorer les requêtes SQL complexes
    # et travailler directement avec les données du modèle
    try:
        # Définir une plage par défaut (derniers 3 mois)
        today = date.today()
        default_min_date = today - relativedelta(months=3)
        default_max_date = today
        
        # Récupérer tous les enregistrements correspondant au domaine
        # et extraire les min/max dates directement des données Python
        records = model.search(domain or [])
        if records:
            date_values = []
            # Extraire les valeurs de date de tous les enregistrements
            for record in records:
                field_value = record[field_name]
                if field_value:
                    # Convertir en date si c'est un datetime
                    if isinstance(field_value, datetime):
                        field_value = field_value.date()
                    date_values.append(field_value)
            
            if date_values:
                min_date = min(date_values)
                max_date = max(date_values)
            else:
                min_date = default_min_date
                max_date = default_max_date
        else:
            # Pas de données - utiliser les dates par défaut
            min_date = default_min_date
            max_date = default_max_date
        
        # Limiter à 1 an maximum pour éviter les plages trop grandes
        if (max_date - min_date).days > 365:
            max_date = min_date + timedelta(days=365)
            _logger.warning("Date range for %s limited to 1 year", field_name)
    except Exception as e:
        _logger.error("Error in _build_date_range for field %s: %s", field_name, e)
        # En cas d'erreur, générer une plage par défaut (3 derniers mois)
        min_date = date.today() - relativedelta(months=3)
        max_date = date.today()
    
    # Generate all intermediate dates based on interval
    date_values = []
    current_date = min_date
    
    if interval == 'day':
        delta = timedelta(days=1)
        format_str = '%Y-%m-%d'
    elif interval == 'week':
        delta = timedelta(weeks=1)
        # Use ISO week format
        format_str = 'W%W %Y'
    elif interval == 'month':
        # For months, use a relative delta
        delta = relativedelta(months=1)
        format_str = '%Y-%m'
    elif interval == 'quarter':
        delta = relativedelta(months=3)
        # Custom handling for quarters
        format_str = 'Q%q %Y'
    elif interval == 'year':
        delta = relativedelta(years=1)
        format_str = '%Y'
    else:
        # Default to month
        delta = relativedelta(months=1)
        format_str = '%Y-%m'
    
    while current_date <= max_date:
        # Format based on interval
        if interval == 'week':
            date_values.append(f"W{current_date.isocalendar()[1]} {current_date.year}")
        elif interval == 'quarter':
            quarter = (current_date.month - 1) // 3 + 1
            date_values.append(f"Q{quarter} {current_date.year}")
        else:
            date_values.append(current_date.strftime(format_str))
        
        # Move to next date
        current_date += delta
    
    return date_values


def _generate_empty_combinations(model, group_by_list, domain, results):
    """Generate all combinations for fields with show_empty=True.
    Takes into account existing values for fields without show_empty.
    """
    # Split fields with and without show_empty
    show_empty_fields = []
    non_show_empty_fields = []
    all_values = {}
    
    # Identifier les champs qui ont des valeurs NULL/None dans les résultats existants
    # pour assurer la cohérence dans le traitement des valeurs NULL
    fields_with_nulls = set()
    
    for gb in group_by_list:
        field = gb.get('field')
        if not field:
            continue
            
        show_empty = gb.get('show_empty', False)
        interval = gb.get('interval')
        
        if show_empty and model._fields[field].type not in ['binary']:
            show_empty_fields.append((field, interval))
            
            if model._fields[field].type in ['date', 'datetime']:
                # Approche plus robuste : utiliser les dates réelles des données existantes
                # et y ajouter les dates récentes pour compléter
                date_values = []
                
                # 1. Utiliser notre propre requête SQL pour obtenir les dates min/max directement
                # à partir de la base de données, indépendamment des résultats intermédiaires
                try:
                    # Trouver les dates min et max réelles dans la base de données
                    min_max_query = f"""
                        SELECT 
                            MIN({field}::date) as min_date,
                            MAX({field}::date) as max_date
                        FROM {model._table}
                        WHERE {field} IS NOT NULL
                    """
                    model.env.cr.execute(min_max_query)
                    date_range = model.env.cr.fetchone()
                    
                    if date_range and date_range[0] and date_range[1]:
                        db_min_date = date_range[0]  # Date minimum de la base
                        db_max_date = date_range[1]  # Date maximum de la base
                        
                        # Filtrer les dates pour n'utiliser que celles qui ont réellement des données
                        # Cela corrige le problème où show_empty génère trop de dates intermédiaires
                        dates_with_data_query = f"""
                            SELECT DISTINCT
                                EXTRACT(YEAR FROM {field}::date) as year,
                                EXTRACT(MONTH FROM {field}::date) as month
                            FROM {model._table}
                            WHERE {field} IS NOT NULL
                            ORDER BY year, month
                        """
                        model.env.cr.execute(dates_with_data_query)
                        dates_with_data = model.env.cr.fetchall()
                        _logger.info("Found %s dates with data for field %s", len(dates_with_data), field)
                        
                        # Utiliser les dates min/max de la base de données pour générer une plage complète
                        # C'est le comportement que l'utilisateur préfère
                        
                        # Assurer que nous avons aussi quelques semaines récentes
                        today = date.today()
                        actual_max_date = max(db_max_date, today)
                        
                        # 2. Générer toutes les valeurs selon l'intervalle
                        if interval == 'week':
                            # Convertir en début de semaine
                            week_min = db_min_date - timedelta(days=db_min_date.weekday())
                            week_max = actual_max_date + timedelta(days=(6 - actual_max_date.weekday()))
                            
                            # Limiter à une année pour éviter les plages trop longues
                            if (week_max - week_min).days > 365:
                                week_min = week_max - timedelta(days=365)
                            
                            # Générer toutes les semaines complètes
                            current = week_min
                            while current <= week_max:
                                week_str = f"W{current.isocalendar()[1]} {current.year}"
                                date_values.append(week_str)
                                current += timedelta(days=7)
                        
                        elif interval == 'month':
                            # Convertir en début de mois
                            month_min = date(db_min_date.year, db_min_date.month, 1)
                            month_max = date(actual_max_date.year, actual_max_date.month, 1)
                            
                            # Limiter à deux ans pour éviter les plages trop longues
                            if (month_max.year - month_min.year) * 12 + (month_max.month - month_min.month) > 24:
                                month_min = date(month_max.year - 2, month_max.month, 1)
                            
                            # Générer tous les mois
                            current = month_min
                            while current <= month_max:
                                # Utiliser le format complet pour correspondre à ce que Odoo renvoie
                                date_values.append(current.strftime('%B %Y'))
                                current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)  # Prochain mois
                        
                        elif interval == 'quarter':
                            # Convertir en début de trimestre
                            q_min = db_min_date.month - 1
                            q_min = q_min - (q_min % 3)
                            quarter_min = date(db_min_date.year, q_min + 1 if q_min > 0 else 1, 1)
                            
                            q_max = actual_max_date.month - 1
                            q_max = q_max - (q_max % 3)
                            quarter_max = date(actual_max_date.year, q_max + 1 if q_max > 0 else 1, 1)
                            
                            # Générer tous les trimestres
                            current = quarter_min
                            while current <= quarter_max:
                                quarter = ((current.month - 1) // 3) + 1
                                date_values.append(f"Q{quarter} {current.year}")
                                current = date(current.year + (1 if current.month > 9 else 0), 
                                               ((current.month - 1 + 3) % 12) + 1, 1)
                        
                        elif interval == 'year':
                            for year in range(db_min_date.year, actual_max_date.year + 1):
                                date_values.append(str(year))
                        
                        else:  # day
                            # Pour les jours, limiter à 60 jours maximum
                            day_max = min(db_max_date, db_min_date + timedelta(days=60))
                            current = db_min_date
                            while current <= day_max:
                                date_values.append(current.strftime('%d %b %Y'))
                                current += timedelta(days=1)
                                
                    else:
                        # Si pas de dates dans la BD, utiliser des dates récentes
                        existing_dates = set()
                        for r in results:
                            if field in r and r[field]:
                                existing_dates.add(r[field])
                        
                        for date_val in existing_dates:
                            if date_val:
                                date_values.append(date_val)
                except Exception as e:
                    _logger.error("Error generating date values: %s", e)
                    # En cas d'erreur, utiliser les dates des résultats
                    existing_dates = set()
                    for r in results:
                        if field in r and r[field]:
                            existing_dates.add(r[field])
                    
                    for date_val in existing_dates:
                        if date_val:
                            date_values.append(date_val)
                
                # 3. Si on n'a toujours pas assez de dates, ajouter des dates récentes
                if len(date_values) < 3:
                    today = date.today()
                    
                    if interval == 'day':
                        # Formatter les dates exactement comme Odoo
                        for i in range(7):
                            dt = today - timedelta(days=i)
                            formatted = dt.strftime('%d %b %Y')  # Format: '11 Apr 2025'
                            if formatted not in date_values:
                                date_values.append(formatted)
                    elif interval == 'week':
                        for i in range(4):
                            week_date = today - timedelta(weeks=i)
                            formatted = f"W{week_date.isocalendar()[1]} {week_date.year}"
                            if formatted not in date_values:
                                date_values.append(formatted)
                    elif interval == 'month':
                        for i in range(6):
                            month_date = today - relativedelta(months=i)
                            formatted = month_date.strftime('%B %Y')  # Format: 'April 2025'
                            if formatted not in date_values:
                                date_values.append(formatted)
                    elif interval == 'quarter':
                        for i in range(4):
                            quarter_date = today - relativedelta(months=i*3)
                            quarter = (quarter_date.month - 1) // 3 + 1
                            formatted = f"Q{quarter} {quarter_date.year}"
                            if formatted not in date_values:
                                date_values.append(formatted)
                    elif interval == 'year':
                        for i in range(3):
                            formatted = str(today.year - i)
                            if formatted not in date_values:
                                date_values.append(formatted)
                
                # Utiliser cette plage fixe
                all_values[(field, interval)] = date_values
            else:
                # Pour tous les autres types de champs
                all_values[(field, interval)] = _get_field_values(model, field, domain)
        else:
            non_show_empty_fields.append((field, interval))
    
    if not show_empty_fields:
        return []
        
    # For fields without show_empty, use only values that exist in results
    existing_values = {}
    for field, interval in non_show_empty_fields:
        field_with_interval = f"{field}:{interval}" if interval else field
        values = set()
        for result in results:
            val = result.get(field_with_interval)
            if val is not None:
                values.add(val)
        existing_values[(field, interval)] = list(values)
    
    # Prepare for all combinations
    all_fields = non_show_empty_fields + show_empty_fields
    all_fields_values = []
    all_field_names = []
    
    for field, interval in all_fields:
        if (field, interval) in existing_values:
            # Field without show_empty - use existing values
            values = existing_values[(field, interval)]
        else:
            # Field with show_empty - use all possible values
            values = all_values[(field, interval)]
            
        if values:  # Only add if there are values
            all_fields_values.append(values)
            all_field_names.append(field)
    
    if not all_fields_values:
        return []
    
    # Générer toutes les combinaisons valides
    # TOUJOURS utiliser des dictionnaires pour la cohérence des retours
    if len(all_fields_values) == 1 and len(all_field_names) == 1:
        # Cas spécial : un seul champ
        field_name = all_field_names[0]
        return [{field_name: value} for value in all_fields_values[0]]
    elif len(all_fields_values) >= 1:
        # Cas normal : plusieurs champs ou combinaisons
        combinations = list(itertools.product(*all_fields_values))
        return [dict(zip(all_field_names, combo)) for combo in combinations]
    else:
        # Cas où aucune valeur n'est trouvée
        return []


def _handle_show_empty(results, model, group_by_list, domain, measures=None):
    """Handle show_empty for groupBy fields by filling in missing combinations."""
    if not any(gb.get('show_empty', False) for gb in group_by_list):
        return results  # No show_empty, return original results
    
    # Generate all possible combinations for show_empty fields
    all_combinations = _generate_empty_combinations(model, group_by_list, domain, results)
    if not all_combinations:
        return results
    
    # Create a dictionary for easy lookup of existing results
    existing_results = {}
    for result in results:
        # Create a key based on groupby field values
        key_parts = []
        for gb in group_by_list:
            field = gb.get('field')
            interval = gb.get('interval')
            if field:
                field_with_interval = f"{field}:{interval}" if interval else field
                value = result.get(field_with_interval)
                
                # Format the value in a consistent way
                if isinstance(value, tuple) and len(value) == 2:
                    # Extract the ID for consistent lookup
                    formatted_value = value[0]
                elif isinstance(value, dict) and 'id' in value:
                    # Extract the ID for consistent lookup
                    formatted_value = value['id']
                else:
                    formatted_value = value
                    
                key_parts.append(str(formatted_value))
        
        existing_results[tuple(key_parts)] = result
    
    # Create combined results with empty values for missing combinations
    combined_results = []
    
    for combo in all_combinations:
        # Create a key to check if this combination exists in results
        key_parts = []
        skip_combo = False
        
        # S'assurer que combo est toujours un dictionnaire à ce stade
        if not isinstance(combo, dict):
            _logger.error("Unexpected non-dict combo in _handle_show_empty: %s", combo)
            continue
            
        for gb in group_by_list:
            field = gb.get('field')
            if field:
                value = combo.get(field, '')
                
                # Skip combinations with None values for date fields that don't have show_empty
                if value is None and not gb.get('show_empty', False):
                    skip_combo = True
                    break
                
                # Format the value in a consistent way
                if isinstance(value, tuple) and len(value) == 2:
                    # Extract the ID for consistent lookup
                    formatted_value = value[0]
                elif isinstance(value, dict) and 'id' in value:
                    # Extract the ID for consistent lookup
                    formatted_value = value['id']
                else:
                    formatted_value = value
                    
                key_parts.append(str(formatted_value))
        
        # Skip this combination if it has None values for non-show_empty fields
        if skip_combo:
            continue
        
        # S'assurer que la clé ne contient pas "None" comme valeur textuelle
        # car ça crée des entrées indésirables
        if "None" in key_parts:
            continue
            
        combo_key = tuple(key_parts)
        
        if combo_key in existing_results:
            # Use existing result
            combined_results.append(existing_results[combo_key])
        else:
            # Create new empty result with correct structure
            new_result = {}
            
            # Add all accumulated measures with default values
            for measure in measures or []:
                field = measure.get('field')
                agg = measure.get('aggregation')
                # Set default value (0 for numeric fields, False for others)
                new_result[field] = 0 if model._fields[field].type in ['float', 'monetary', 'integer'] else False
            
            # Add combination values to result, avec les formats compatibles read_group
            for gb in group_by_list:
                field = gb.get('field')
                interval = gb.get('interval')
                
                if field in combo:
                    # Ajouter à la fois le champ original et le champ avec intervalle
                    # pour assurer la compatibilité avec _transform_graph_data
                    new_result[field] = combo[field]
                    
                    # Ajouter également avec le format field:interval pour assurer la compatibilité
                    if interval:
                        field_with_interval = f"{field}:{interval}"
                        new_result[field_with_interval] = combo[field]
                
            combined_results.append(new_result)
    
    return combined_results


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
        
        # Process secondary fields and measures if they exist
        if secondary_fields:
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
                        formatted_sec_value = sec_value['display_name'] # display name for cleaner output
                    
                    # Construct the key for this measure and secondary field value
                    measure_key = f"{field}|{formatted_sec_value}" if sec_field else field
                    
                    # Get the measure value from the result
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
                
                # Get the measure value from the result
                if agg == 'count':
                    measure_value = result.get('__count', 0)
                else:
                    measure_value = result.get(field, 0)
                
                # Add to the primary group
                primary_groups[dict_key][field] = measure_value
    
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
                        # Traitement spécial pour les dates au format "DD MMM YYYY" (ex: "11 Apr 2025")
                        if ' ' in key and not key.startswith('W') and not key.startswith('Q'):
                            try:
                                parts = key.split(' ')
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
                                
                                # Format "DD MMM YYYY" (ex: "11 Apr 2025")
                                if len(parts) == 3 and parts[1] in month_map:
                                    day_num = int(parts[0])
                                    month_num = month_map[parts[1]]
                                    year_num = int(parts[2])
                                    date_obj = datetime(year_num, month_num, day_num)
                                    _logger.info("Key: %s => Date value: %s", key, date_obj)
                                    return date_obj
                                # Format "MMM YYYY" (ex: "Apr 2025")
                                elif len(parts) == 2 and parts[0] in month_map:
                                    month_num = month_map[parts[0]]
                                    year_num = int(parts[1])
                                    # Créer la date du premier jour du mois
                                    date_obj = datetime(year_num, month_num, 1)
                                    _logger.info("Key: %s => Date value: %s", key, date_obj)
                                    return date_obj
                            except Exception as e:
                                _logger.error("Failed to parse date format %s: %s", key, e)
                        
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
                                date_obj = first_day + timedelta(days=(week_num-1)*7)
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
