"""
Constants useful across modules.
"""

#: GTFS data types
DTYPES = {
    "agency": {
        "agency_id": "string",
        "agency_name": "string",
        "agency_url": "string",
        "agency_timezone": "string",
        "agency_lang": "string",
        "agency_phone": "string",
        "agency_fare_url": "string",
        "agency_email": "string",
    },
    "attributions": {
        "attribution_id": "string",
        "agency_id": "string",
        "route_id": "string",
        "trip_id": "string",
        "organization_name": "string",
        "is_producer": "Int32",
        "is_operator": "Int32",
        "is_authority": "Int32",
        "attribution_url": "string",
        "attribution_email": "string",
        "attribution_phone": "string",
    },
    "calendar": {
        "service_id": "string",
        "monday": "Int32",
        "tuesday": "Int32",
        "wednesday": "Int32",
        "thursday": "Int32",
        "friday": "Int32",
        "saturday": "Int32",
        "sunday": "Int32",
        "start_date": "string",
        "end_date": "string",
    },
    "calendar_dates": {
        "service_id": "string",
        "date": "string",
        "exception_type": "Int32",
    },
    "fare_attributes": {
        "fare_id": "string",
        "price": "float",
        "currency_type": "string",
        "payment_method": "Int32",
        "transfers": "Int32",
        "transfer_duration": "Int16",
    },
    "fare_rules": {
        "fare_id": "string",
        "route_id": "string",
        "origin_id": "string",
        "destination_id": "string",
        "contains_id": "string",
    },
    "feed_info": {
        "feed_publisher_name": "string",
        "feed_publisher_url": "string",
        "feed_lang": "string",
        "feed_start_date": "string",
        "feed_end_date": "string",
        "feed_version": "string",
    },
    "frequencies": {
        "trip_id": "string",
        "start_time": "string",
        "end_time": "string",
        "headway_secs": "Int16",
        "exact_times": "Int32",
    },
    "routes": {
        "route_id": "string",
        "agency_id": "string",
        "route_short_name": "string",
        "route_long_name": "string",
        "route_desc": "string",
        "route_type": "Int32",
        "route_url": "string",
        "route_color": "string",
        "route_text_color": "string",
    },
    "shapes": {
        "shape_id": "string",
        "shape_pt_lat": "float",
        "shape_pt_lon": "float",
        "shape_pt_sequence": "Int32",
        "shape_dist_traveled": "float",
    },
    "stop_times": {
        "trip_id": "string",
        "arrival_time": "string",
        "departure_time": "string",
        "stop_id": "string",
        "stop_sequence": "Int32",
        "stop_headsign": "string",
        "pickup_type": "Int32",
        "drop_off_type": "Int32",
        "shape_dist_traveled": "float",
        "timepoint": "Int32",
    },
    "stops": {
        "stop_id": "string",
        "stop_code": "string",
        "stop_name": "string",
        "stop_desc": "string",
        "stop_lat": "float",
        "stop_lon": "float",
        "zone_id": "string",
        "stop_url": "string",
        "location_type": "Int32",
        "parent_station": "string",
        "stop_timezone": "string",
        "wheelchair_boarding": "Int32",
    },
    "transfers": {
        "from_stop_id": "string",
        "to_stop_id": "string",
        "transfer_type": "Int32",
        "min_transfer_time": "Int16",
    },
    "trips": {
        "route_id": "string",
        "service_id": "string",
        "trip_id": "string",
        "trip_headsign": "string",
        "trip_short_name": "string",
        "direction_id": "Int32",
        "block_id": "string",
        "shape_id": "string",
        "wheelchair_accessible": "Int32",
        "bikes_allowed": "Int32",
    },
}

#: ID fields for merging
ID_FIELDS = [
    "agency_id",
    "attribution_id",
    "stop_id",
    "route_id",
    "trip_id",
    "service_id",
    "shape_id",
    "fare_id",
    "zone_id",
    "origin_id",
    "destination_id",
    "parent_station",
    "from_stop_id",
    "to_stop_id",
    "block_id",
]

#: ID to (file, id_col) for ID remapping
ID_REMAP = {
    "stop_id": [
        ("stops", "stop_id"),
        ("stop_times", "stop_id"),
        ("transfers", "from_stop_id"),
        ("transfers", "to_stop_id"),
        ("stops", "parent_station"),
    ],
    "route_id": [
        ("routes", "route_id"),
        ("trips", "route_id"),
        ("fare_rules", "route_id"),
        ("attributions", "route_id"),
    ],
    "service_id": [
        ("calendar", "service_id"),
        ("trips", "service_id"),
        ("calendar_dates", "service_id"),
    ],
}

#: Valid distance units
DIST_UNITS = ["ft", "mi", "m", "km"]

#: Feed attributes
FEED_ATTRS = [
    "agency",
    "attributions",
    "calendar",
    "calendar_dates",
    "fare_attributes",
    "fare_rules",
    "feed_info",
    "frequencies",
    "routes",
    "shapes",
    "stops",
    "stop_times",
    "trips",
    "transfers",
    "dist_units",
]

#: WGS84 coordinate reference system for Geopandas
WGS84 = "EPSG:4326"

#: Colorbrewer 8-class Set2 colors
COLORS_SET2 = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]
