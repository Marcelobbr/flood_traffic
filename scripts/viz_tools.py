import altair as alt

def chart_with_rule(source, y, selection):

    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                        fields=['time'], empty='none')
    
    color = alt.condition(selection,
                          alt.Color('city:N', legend=None),
                          alt.value('lightgray'))
    
    line = alt.Chart(source).mark_line().encode(
        x='time:T', 
        y=y, 
        color=color
    ).properties(width=800, height=300).transform_filter(selection)
    
    selectors = alt.Chart(source).mark_point().encode(
        x='time:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points = line.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )
    text = line.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, y, alt.value(' ')),
        color=alt.value('black')
    )
    
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='time:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    chart = alt.layer(
                line, selectors, points, rules, text
            ).properties(
                width=800, height=300
            )
    
    return chart

def wind_rain_interactive_chart(source, grid=True, title=''):
    # Create a selection that chooses the nearest point & selects based on x-value
    nearest = alt.selection(type='single', nearest=True, on='mouseover',
                            fields=['time'], empty='none')

    # The basic line

    line1 = alt.Chart(source).mark_line().encode(x='time:T', y='precipIntensity:Q')
    line2 = alt.Chart(source).mark_line().encode(x='time:T', y='windSpeed:Q', color=alt.value('red'))

    line = (line1 + line2).properties(title=title).resolve_scale(y='independent')

    # line = alt.Chart(source).mark_line(interpolate='basis').encode(
    #     x='x:Q',
    #     y='y:Q',
    #     color='category:N'
    # )

    # Transparent selectors across the chart. This is what tells us
    # the x-value of the cursor
    selectors = alt.Chart(source).mark_point().encode(
        x='time:T',
        opacity=alt.value(0),
    ).add_selection(
        nearest
    )

    # Draw points on the line, and highlight based on selection
    points1 = line1.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    points2 = line2.mark_point().encode(
        opacity=alt.condition(nearest, alt.value(1), alt.value(0))
    )

    points = (points1 + points2).resolve_scale(y='independent')

    # Draw text labels near the points, and highlight based on selection
    text1 = line1.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'precipIntensity:Q', alt.value(' ')),
        color=alt.value('black')
    )

    text2 = line2.mark_text(align='left', dx=5, dy=-5).encode(
        text=alt.condition(nearest, 'windSpeed:Q', alt.value(' ')),
        color=alt.value('black')
    )

    text = (text1 + text2).resolve_scale(y='independent')

    # Draw a rule at the location of the selection
    rules = alt.Chart(source).mark_rule(color='gray').encode(
        x='time:T',
    ).transform_filter(
        nearest
    )

    # Put the five layers into a chart and bind the data
    chart = alt.layer(
                line, selectors, points, rules, text
            ).properties(
                width=800, height=300
            )

    if not grid:
        return chart.configure_axis(grid=False)
    return chart