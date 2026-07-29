WITH seven_day_users AS (
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210101,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
    
    UNION
    
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210102,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
    
    UNION
    
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210103,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
    
    UNION
    
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210104,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
    
    UNION
    
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210105,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
    
    UNION
    
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210106,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
    
    UNION
    
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210107,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
),
two_day_users AS (
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210106,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
    
    UNION
    
    SELECT DISTINCT "USER_PSEUDO_ID"
    FROM GA4.GA4_OBFUSCATED_SAMPLE_ECOMMERCE.EVENTS_20210107,
    LATERAL FLATTEN(input => "EVENT_PARAMS") ep
    WHERE ep.value:key::STRING = 'engagement_time_msec'
    AND ep.value:value:int_value::INT > 0
)
SELECT COUNT(*) AS distinct_pseudo_users
FROM seven_day_users s
WHERE s."USER_PSEUDO_ID" NOT IN (SELECT t."USER_PSEUDO_ID" FROM two_day_users t)