/* Expedia hotel events -> LoopAd SDK-like raw_events.

   This transform is deterministic. It does not invent random behavior.
   Every emitted raw event is derived from an expedia_hotel_events row:

   - hotel_search: every Expedia search row
   - hotel_detail_view: every row with a hotel market/cluster candidate
   - hotel_click: rows with repeated result interactions (cnt >= 2) or booking
   - booking_start: booking rows, plus high-intent non-booking rows (cnt >= 4)
   - booking_complete: only rows where is_booking = 1

   properties_json intentionally carries Expedia source fields and derived
   SDK-like properties. Derived values are marked with *_source fields.
*/

INSERT INTO raw_events (
    project_id,
    write_key,
    schema_version,
    event_id,
    event_name,
    event_time,
    received_at,
    source,
    user_id,
    session_id,
    properties_json,
    validation_status
)
-- TRANSFORM_SELECT_START
WITH
    parseDateTimeBestEffortOrNull({start_datetime:String}) AS start_ts,
    parseDateTimeBestEffortOrNull({end_datetime:String}) AS end_ts,
    if({user_sample_modulo:UInt64} = 0, 1, {user_sample_modulo:UInt64}) AS sample_modulo,
    {user_sample_remainder:UInt64} AS sample_remainder,
    if(
        {max_source_rows:UInt64} = 0,
        toUInt64(18446744073709551615),
        {max_source_rows:UInt64}
    ) AS source_row_limit
SELECT
    {project_id:String} AS project_id,
    {write_key:String} AS write_key,
    {schema_version:String} AS schema_version,
    concat('evt_expedia_', lower(hex(cityHash64(concat(source_row_hash, '|', event_name))))) AS event_id,
    event_name,
    addSeconds(toDateTime64(date_time, 3, 'UTC'), event_offset_seconds) AS event_time,
    addSeconds(toDateTime64(date_time, 3, 'UTC'), event_offset_seconds + 1) AS received_at,
    {source:String} AS source,
    concat('expedia-user-', toString(user_id)) AS user_id,
    session_id,
    toJSONString(map(
        'source_dataset', 'expedia_hotel_events',
        'source_row_hash', source_row_hash,
        'source_event_step', event_step,
        'source_event_rule', event_rule,
        'expedia_user_id', toString(user_id),
        'site_name', toString(site_name),
        'posa_continent', toString(posa_continent),
        'user_location_country', toString(user_location_country),
        'user_location_region', toString(user_location_region),
        'user_location_city', toString(user_location_city),
        'is_mobile', toString(is_mobile),
        'is_package', toString(is_package),
        'channel', toString(channel),
        'destination_id', toString(srch_destination_id),
        'destination_name', concat('expedia destination ', toString(srch_destination_id)),
        'destination_type_id', toString(srch_destination_type_id),
        'hotel_continent', toString(hotel_continent),
        'hotel_country', toString(hotel_country),
        'hotel_market', toString(hotel_market),
        'hotel_cluster', toString(hotel_cluster),
        'hotel_city', '',
        'checkin_date', checkin_date,
        'checkout_date', checkout_date,
        'booking_window_days', toString(booking_window_days),
        'stay_nights', toString(stay_nights),
        'adult_count', toString(srch_adults_cnt),
        'child_count', toString(srch_children_cnt),
        'rooms', toString(srch_rm_cnt),
        'traveler_count', toString(traveler_count),
        'room_type', room_type,
        'preferred_category', preferred_category,
        'orig_destination_distance', orig_destination_distance_text,
        'is_booking_row', toString(is_booking),
        'cnt', toString(cnt),
        'deal', deal,
        'deal_source', deal_source,
        'free_cancellation', free_cancellation,
        'free_cancellation_source', free_cancellation_source,
        'breakfast_included', breakfast_included,
        'breakfast_included_source', breakfast_included_source,
        'price', toString(estimated_price),
        'estimated_price', toString(estimated_price),
        'price_source', 'derived_from_expedia_fields',
        'revenue', if(is_booking = 1, toString(estimated_price * greatest(stay_nights, 1)), '0'),
        'page_path', page_path,
        'landing_url', page_path,
        'session_id', session_id
    )) AS properties_json,
    'valid' AS validation_status
FROM (
    SELECT
        *,
        tupleElement(event_tuple, 1) AS event_name,
        tupleElement(event_tuple, 2) AS event_offset_seconds,
        tupleElement(event_tuple, 3) AS event_step,
        tupleElement(event_tuple, 4) AS event_rule,
        multiIf(
            tupleElement(event_tuple, 1) IN ('page_view', 'hotel_search'),
                concat('/hotels/search?destination_id=', toString(srch_destination_id)),
            tupleElement(event_tuple, 1) IN ('hotel_detail_view', 'hotel_click'),
                concat('/hotels/', toString(hotel_country), '/', toString(hotel_market), '/', toString(hotel_cluster)),
            tupleElement(event_tuple, 1) = 'booking_start',
                '/checkout/start',
            tupleElement(event_tuple, 1) = 'booking_complete',
                '/checkout/complete',
            '/hotels'
        ) AS page_path
    FROM (
        SELECT
            *,
            arrayFilter(
                event -> tupleElement(event, 1) != '',
                [
                    tuple('page_view', -2, 'search_page_view', 'expedia_row_seen'),
                    tuple('hotel_search', 0, 'hotel_search', 'expedia_search_row'),
                    tuple('hotel_detail_view', 5, 'hotel_detail_view', 'hotel_candidate_present'),
                    tuple(if(cnt >= 2 OR is_booking = 1, 'hotel_click', ''), 10, 'hotel_click', 'cnt_or_booking_intent'),
                    tuple(if(is_booking = 1 OR (is_booking = 0 AND cnt >= 4), 'booking_start', ''), 20, 'booking_start', 'booking_or_high_cnt_abandonment'),
                    tuple(if(is_booking = 1, 'booking_complete', ''), 40, 'booking_complete', 'observed_booking')
                ]
            ) AS event_tuples
        FROM (
            SELECT
                date_time,
                site_name,
                posa_continent,
                user_location_country,
                user_location_region,
                user_location_city,
                orig_destination_distance,
                user_id,
                is_mobile,
                is_package,
                channel,
                srch_ci,
                srch_co,
                srch_adults_cnt,
                srch_children_cnt,
                srch_rm_cnt,
                srch_destination_id,
                srch_destination_type_id,
                hotel_continent,
                hotel_country,
                hotel_market,
                is_booking,
                cnt,
                hotel_cluster,
                if(isNull(srch_ci), '', toString(assumeNotNull(srch_ci))) AS checkin_date,
                if(isNull(srch_co), '', toString(assumeNotNull(srch_co))) AS checkout_date,
                if(
                    isNull(srch_ci) OR isNull(srch_co),
                    0,
                    greatest(dateDiff('day', assumeNotNull(srch_ci), assumeNotNull(srch_co)), 0)
                ) AS stay_nights,
                if(
                    isNull(srch_ci),
                    0,
                    greatest(dateDiff('day', toDate(date_time), assumeNotNull(srch_ci)), 0)
                ) AS booking_window_days,
                srch_adults_cnt + srch_children_cnt AS traveler_count,
                if(
                    isNull(orig_destination_distance),
                    '',
                    toString(assumeNotNull(orig_destination_distance))
                ) AS orig_destination_distance_text,
                concat(
                    'src_',
                    lower(hex(cityHash64(concat(
                        toString(date_time), '|',
                        toString(user_id), '|',
                        toString(srch_destination_id), '|',
                        toString(hotel_country), '|',
                        toString(hotel_market), '|',
                        toString(hotel_cluster), '|',
                        toString(is_booking), '|',
                        toString(cnt)
                    ))))
                ) AS source_row_hash,
                concat(
                    'sess_expedia_',
                    toString(user_id),
                    '_',
                    lower(hex(cityHash64(concat(
                        toString(toDate(date_time)), '|',
                        toString(srch_destination_id), '|',
                        toString(hotel_market)
                    ))))
                ) AS session_id,
                multiIf(
                    srch_rm_cnt >= 2, 'multi_room',
                    srch_children_cnt > 0, 'family_room',
                    srch_adults_cnt = 1 AND srch_children_cnt = 0, 'single_room',
                    'standard_room'
                ) AS room_type,
                multiIf(
                    srch_children_cnt > 0, 'family_travel',
                    is_package = 1, 'package_travel',
                    srch_adults_cnt = 1 AND stay_nights <= 2, 'business_or_solo_travel',
                    booking_window_days <= 7, 'last_minute_travel',
                    'leisure_travel'
                ) AS preferred_category,
                multiIf(
                    is_package = 1, 'package_bundle',
                    booking_window_days >= 30, 'early_booking',
                    cnt >= 4, 'comparison_deal',
                    ''
                ) AS deal,
                multiIf(
                    is_package = 1, 'is_package',
                    booking_window_days >= 30, 'booking_window_days',
                    cnt >= 4, 'cnt_high_intent',
                    ''
                ) AS deal_source,
                if(booking_window_days >= 14 OR is_package = 1, '1', '0') AS free_cancellation,
                'inferred_from_booking_window_or_package' AS free_cancellation_source,
                if(is_package = 1 AND stay_nights >= 2, '1', '0') AS breakfast_included,
                'inferred_from_package_and_stay_nights' AS breakfast_included_source,
                toUInt32(
                    greatest(
                        45000,
                        65000
                        + toInt32(hotel_continent) * 7000
                        + toInt32(modulo(hotel_country, 80)) * 900
                        + toInt32(modulo(hotel_market, 120)) * 450
                        + toInt32(greatest(srch_rm_cnt, 1)) * 22000
                        + toInt32(greatest(srch_adults_cnt, 1)) * 9000
                        + toInt32(srch_children_cnt) * 6000
                        + if(is_package = 1, -12000, 0)
                        + if(booking_window_days <= 7, 18000, 0)
                    )
                ) AS estimated_price
            FROM expedia_hotel_events
            WHERE (isNull(start_ts) OR date_time >= start_ts)
              AND (isNull(end_ts) OR date_time < end_ts)
              AND modulo(cityHash64(toString(user_id)), sample_modulo) = sample_remainder
            ORDER BY cityHash64(concat(
                toString(user_id), '|',
                toString(date_time), '|',
                toString(srch_destination_id), '|',
                toString(hotel_market), '|',
                toString(hotel_cluster)
            ))
            LIMIT source_row_limit
        ) AS source_rows
    ) AS rows_with_events
    ARRAY JOIN event_tuples AS event_tuple
) AS expanded_events
-- TRANSFORM_SELECT_END
;
