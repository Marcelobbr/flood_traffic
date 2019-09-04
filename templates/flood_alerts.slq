WITH p AS
         (SELECT uuid, street, MAX(nthumbsup) + 1 AS interactions, arbitrary(latitude) as latitude,
          arbitrary(longitude) as longitude, MAX(reliability) as reliability, MIN(pub_utc_date) as start_time,
          MAX(pub_utc_date) as end_time 
         FROM {table}
         WHERE ({date_filter})
              AND city = '{city}'
              AND subtype = 'HAZARD_WEATHER_FLOOD'
         GROUP BY  uuid, street),
total AS
        (SELECT SUM(interactions) as interactions FROM p)
     
SELECT uuid, latitude, longitude, p.interactions, street, reliability, start_time, end_time,
    ROUND(CAST(p.interactions AS double) / total.interactions, 4) AS share,
    SUM(CAST(p.interactions AS double)) OVER (ORDER BY p.interactions DESC, reliability DESC, start_time) / total.interactions AS cum_share
FROM p, total
ORDER BY interactions DESC