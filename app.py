
# HOTELWISE — Application foundation

import streamlit as st
import joblib
from pathlib import Path
import pandas as pd
import requests

# Page configuration
st.set_page_config(
    page_title="HotelWise | Revenue Intelligence",
    page_icon="🏨",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load saved models and analysis
@st.cache_resource
def load_artifacts():
    file_path = Path(__file__).parent / "hotelwise_artifacts.pkl"
    return joblib.load(file_path)

try:
    artifacts = load_artifacts()
except Exception as e:
    st.error(f"Unable to load HotelWise data: {e}")
    st.stop()

# Retrieve project components
model = artifacts["model"]
logistic_model = artifacts["logistic_model"]
tree_model = artifacts["tree_model"]
rf_model = artifacts["rf_model"]

model_results = artifacts["model_results"]
booking_data = artifacts["booking_data"]
daily_summary = artifacts["daily_summary"]
overbooking_results = artifacts["overbooking_results"]

recommendations = artifacts["booking_recommendations"]
financial_results = artifacts["financial_results"]

hotel_capacities = artifacts["hotel_capacities"]
average_revenue = artifacts["average_revenue_by_hotel"]
relocation_cost = artifacts["relocation_cost"]
# HOTELWISE AI — OLLAMA CONNECTION
def ask_hotelwise_ai(question, context):

    prompt = f"""
    You are HotelWise, an AI decision-support assistant for hotel managers.

    Use the following HotelWise project data to answer the manager.
    Be clear, practical and concise.
    Never invent figures or claim that assumptions are real observations.
    If the data does not contain the answer, say so.

    HOTELWISE DATA:
    {context}

    MANAGER QUESTION:
    {question}
    """

    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": "llama3.2",
            "messages": [
                {"role": "user", "content": prompt}
            ],
            "stream": False
        },
        timeout=120
    )

    response.raise_for_status()

    return response.json()["message"]["content"]


# HotelWise — Premium visual identity

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Playfair+Display:wght@600;700&display=swap');

.stApp {
    background: #F7F6F2;
    color: #202D3D;
    font-family: 'DM Sans', sans-serif;
}

[data-testid="stSidebar"] {
    background: #142538;
    border-right: 1px solid #26394C;
}

[data-testid="stSidebar"] * {
    color: #F5F1E8;
}

[data-testid="stSidebar"] h1 {
    font-family: 'Playfair Display', serif;
    font-size: 30px;
}

[data-testid="stSidebar"] [data-testid="stRadio"] label {
    padding: 9px 10px;
    border-radius: 8px;
}

.main-title {
    font-family: 'Playfair Display', serif;
    font-size: 43px;
    font-weight: 700;
    color: #142538;
    margin-bottom: 0;
}

.subtitle {
    color: #78828B;
    font-size: 15px;
    margin-top: 4px;
    margin-bottom: 25px;
}

h2, h3 {
    color: #142538 !important;
    font-weight: 700 !important;
}

div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #EAE6DC;
    border-radius: 14px;
    padding: 20px 22px;
    box-shadow: 0 4px 16px rgba(20, 37, 56, 0.04);
}

div[data-testid="stMetricLabel"] {
    color: #7B8490;
    font-size: 13px;
    font-weight: 600;
}

div[data-testid="stMetricValue"] {
    color: #142538;
    font-size: 30px;
    font-weight: 700;
}

div[data-testid="stDataFrame"] {
    background: white;
    border-radius: 12px;
}

.stButton > button {
    background: #B99A62;
    color: white;
    border: none;
    border-radius: 8px;
    padding: 10px 20px;
    font-weight: 600;
}

.stButton > button:hover {
    background: #A48650;
    color: white;
}

hr {
    border-color: #E8E3D9;
}

</style>
""", unsafe_allow_html=True)

# Sidebar navigation
with st.sidebar:
    st.markdown("# 🏨 HotelWise")
    st.caption("REVENUE INTELLIGENCE PLATFORM")
    st.divider()

    page = st.radio(
        "NAVIGATION",
        [
            "Executive Overview",
            "Booking Intelligence",
            "Predictive Analytics",
            "Overbooking Simulator",
            "Financial Impact",
            "HotelWise AI Assistant"
        ]
    )

    st.divider()
    st.caption("AI-powered decision support")
    st.caption("Hotel Booking Demand Dataset")

# Main page header
st.markdown('<p class="main-title">HotelWise</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtitle">Turning booking uncertainty into smarter hotel decisions.</p>',
    unsafe_allow_html=True
)

st.divider()


# Executive Overview

if page == "Executive Overview":
    import plotly.express as px

    st.markdown("## Executive Overview")
    st.write(
        "A strategic view of hotel bookings, cancellation behavior "
        "and potential capacity risks."
    )

    # KEY PERFORMANCE INDICATORS
    total_bookings = len(booking_data)
    actual_cancellation_rate = booking_data["actual_cancellation"].mean() * 100
    predicted_cancellation_rate = booking_data["cancellation_probability"].mean() * 100
    number_of_risky_dates = len(overbooking_results)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total Bookings", f"{total_bookings:,}")
    col2.metric("Actual Cancellation Rate", f"{actual_cancellation_rate:.1f}%")
    col3.metric("Predicted Cancellation Rate", f"{predicted_cancellation_rate:.1f}%")
    col4.metric("Dates Requiring Risk Review", number_of_risky_dates)

    st.markdown("---")

    # CANCELLATION PERFORMANCE
    st.markdown("### Cancellation Performance by Hotel")

    hotel_performance = (
        booking_data.groupby("hotel")
        .agg(
            total_bookings=("actual_cancellation", "count"),
            actual_rate=("actual_cancellation", "mean"),
            predicted_rate=("cancellation_probability", "mean")
        )
        .reset_index()
    )

    hotel_performance["actual_rate"] *= 100
    hotel_performance["predicted_rate"] *= 100

    chart_data = hotel_performance.melt(
        id_vars="hotel",
        value_vars=["actual_rate", "predicted_rate"],
        var_name="measure",
        value_name="cancellation_rate"
    )

    chart_data["measure"] = chart_data["measure"].map({
        "actual_rate": "Actual cancellation",
        "predicted_rate": "Predicted cancellation"
    })

    fig = px.bar(
        chart_data,
        x="hotel",
        y="cancellation_rate",
        color="measure",
        barmode="group",
        text_auto=".1f",
        color_discrete_map={
            "Actual cancellation": "#142538",
            "Predicted cancellation": "#B99A62"
        },
        template="simple_white",
        labels={
            "hotel": "Hotel",
            "cancellation_rate": "Cancellation rate (%)",
            "measure": ""
        }
    )

    fig.update_traces(
        textposition="outside",
        marker_line_width=0,
        hovertemplate="<b>%{x}</b><br>%{fullData.name}: %{y:.1f}%<extra></extra>"
    )

    fig.update_layout(
        title="Actual vs Predicted Cancellation Rate",
        title_font=dict(size=19, color="#142538"),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#ECEFF3",
            ticksuffix="%",
            rangemode="tozero"
        ),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(l=20, r=20, t=90, b=20),
        height=400
    )

    st.plotly_chart(fig, use_container_width=True)

    # HOTEL PERFORMANCE TABLE
    st.markdown("### Hotel Performance Summary")

    display_table = hotel_performance.copy()

    display_table.columns = [
        "Hotel",
        "Total bookings",
        "Actual cancellation (%)",
        "Predicted cancellation (%)"
    ]

    display_table["Actual cancellation (%)"] = (
        display_table["Actual cancellation (%)"].map(lambda x: f"{x:.2f}%")
    )

    display_table["Predicted cancellation (%)"] = (
        display_table["Predicted cancellation (%)"].map(lambda x: f"{x:.2f}%")
    )

    st.dataframe(
        display_table,
        use_container_width=True,
        hide_index=True
    )

    # DATA SOURCE
    st.caption(
        "Data source: Hotel Booking Demand dataset. "
        "Predictions are generated by the calibrated Random Forest model."
    )


elif page == "Booking Intelligence":

    import plotly.express as px

    st.markdown("## Booking Intelligence")
    st.write(
        "Explore booking patterns and identify the factors "
        "associated with hotel cancellations."
    )

    st.markdown("### Booking Volume by Hotel")

    hotel_counts = (
        booking_data["hotel"]
        .value_counts()
        .rename_axis("Hotel")
        .reset_index(name="Bookings")
    )

    fig1 = px.pie(
        hotel_counts,
        names="Hotel",
        values="Bookings",
        hole=0.65,
        color_discrete_sequence=["#142538", "#B99A62"],
        template="simple_white"
    )

    fig1.update_traces(
        textposition="inside",
        textinfo="percent+label",
        hovertemplate="<b>%{label}</b><br>Bookings: %{value:,}<extra></extra>"
    )

    fig1.update_layout(
        height=350,
        showlegend=False,
        margin=dict(t=30, b=20, l=20, r=20)
    )

    st.plotly_chart(fig1, use_container_width=True)

    st.markdown("### Cancellation by Booking Lead Time")

    lead_data = booking_data.copy()

    lead_data["Lead Time Group"] = pd.cut(
        lead_data["lead_time"],
        bins=[-1, 30, 90, 180, float("inf")],
        labels=["0–30 days", "31–90 days", "91–180 days", "180+ days"]
    )

    lead_summary = (
        lead_data.groupby("Lead Time Group", observed=False)
        .agg(
            Bookings=("actual_cancellation", "count"),
            Cancellation_Rate=("actual_cancellation", "mean")
        )
        .reset_index()
    )

    lead_summary["Cancellation_Rate"] *= 100

    fig2 = px.bar(
        lead_summary,
        x="Lead Time Group",
        y="Cancellation_Rate",
        text="Cancellation_Rate",
        template="simple_white",
        color_discrete_sequence=["#B99A62"]
    )

    fig2.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="Lead time: %{x}<br>Cancellation: %{y:.1f}%<extra></extra>"
    )

    fig2.update_layout(
        title="Cancellation Rate by Lead Time",
        xaxis_title="Time between booking and arrival",
        yaxis_title="Cancellation rate (%)",
        yaxis=dict(showgrid=False, ticksuffix="%"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### Cancellation by Customer Type")

    customer_summary = (
        booking_data.groupby("customer_type")
        .agg(
            Bookings=("actual_cancellation", "count"),
            Cancellation_Rate=("actual_cancellation", "mean")
        )
        .reset_index()
        .sort_values("Cancellation_Rate", ascending=False)
    )

    customer_summary["Cancellation_Rate"] *= 100

    fig3 = px.bar(
        customer_summary,
        x="customer_type",
        y="Cancellation_Rate",
        text="Cancellation_Rate",
        color_discrete_sequence=["#142538"],
        template="simple_white"
    )

    fig3.update_traces(
        texttemplate="%{text:.1f}%",
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Cancellation: %{y:.1f}%<extra></extra>"
    )

    fig3.update_layout(
        title="Cancellation Rate by Customer Type",
        xaxis_title="Customer type",
        yaxis_title="Cancellation rate (%)",
        yaxis=dict(showgrid=False, ticksuffix="%"),
        xaxis=dict(showgrid=False),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380
    )

    st.plotly_chart(fig3, use_container_width=True)

    st.caption(
        "Analysis is based on the chronological test dataset. "
        "These relationships describe patterns and do not establish causation."
    )


elif page == "Predictive Analytics":

    import plotly.express as px

    st.markdown("## Predictive Analytics")
    st.write(
        "Evaluate the performance of machine-learning models "
        "and understand the cancellation risks predicted by HotelWise."
    )

    # MODEL COMPARISON
    st.markdown("### Model Performance Comparison")

    st.dataframe(
        model_results,
        use_container_width=True,
        hide_index=True
    )

    # MODEL PERFORMANCE CHART
    results_chart = model_results.copy()

    metric_columns = [
        col for col in results_chart.columns
        if col.lower().replace("-", "").replace("_", "").replace(" ", "")
        in ["accuracy", "precision", "recall", "f1", "f1score", "rocauc"]
    ]

    if metric_columns:
        results_long = results_chart.melt(
            id_vars=[results_chart.columns[0]],
            value_vars=metric_columns,
            var_name="Metric",
            value_name="Score"
        )

        fig1 = px.bar(
            results_long,
            x=results_chart.columns[0],
            y="Score",
            color="Metric",
            barmode="group",
            template="simple_white",
            color_discrete_sequence=[
                "#142538", "#B99A62", "#8C9BAA",
                "#C8BBA3", "#526D82", "#A48650"
            ],
            title="Machine Learning Model Comparison"
        )

        fig1.update_layout(
            yaxis=dict(range=[0, 1], showgrid=True, gridcolor="#ECEFF3"),
            xaxis_title="Model",
            yaxis_title="Performance score",
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=430,
            legend_title=""
        )

        st.plotly_chart(fig1, use_container_width=True)

    st.markdown("---")

    # CANCELLATION PROBABILITY DISTRIBUTION
    st.markdown("### Predicted Cancellation Risk")

    fig2 = px.histogram(
        booking_data,
        x="cancellation_probability",
        nbins=30,
        color_discrete_sequence=["#B99A62"],
        template="simple_white",
        title="Distribution of Predicted Cancellation Probabilities"
    )

    fig2.update_layout(
        xaxis_title="Predicted cancellation probability",
        yaxis_title="Number of bookings",
        xaxis=dict(ticksuffix="", showgrid=False),
        yaxis=dict(showgrid=True, gridcolor="#ECEFF3"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=380
    )

    st.plotly_chart(fig2, use_container_width=True)

    st.caption(
        "The probability distribution shows the risk scores assigned "
        "by the calibrated Random Forest model to test bookings. "
        "A higher score indicates a higher predicted cancellation risk."
    )


elif page == "Overbooking Simulator":

    import numpy as np
    import plotly.express as px

    st.markdown("## What-If Decision Simulator")
    st.write(
        "Explore different booking scenarios and discover how many "
        "additional reservations a hotel can accept while controlling "
        "its overcapacity risk."
    )

    # --------------------------------------------------
    # 1. SELECT HOTEL AND HISTORICAL REFERENCE DATE
    # --------------------------------------------------

    st.markdown("### 1. Select a hotel and reference date")

    hotel_options = sorted(
        daily_summary["hotel"].dropna().unique()
    )

    selected_hotel = st.selectbox(
        "Select a hotel",
        hotel_options,
        key="whatif_hotel"
    )

    hotel_dates = daily_summary[
        daily_summary["hotel"] == selected_hotel
    ].copy()

    hotel_dates["arrival_date"] = pd.to_datetime(
        hotel_dates["arrival_date"]
    )

    hotel_dates["_date_key"] = (
        hotel_dates["arrival_date"].dt.strftime("%Y-%m-%d")
    )

    date_options = sorted(
        hotel_dates["_date_key"].dropna().unique()
    )

    selected_date = st.selectbox(
        "Select a historical reference date",
        date_options,
        key="whatif_date"
    )

    selected_row = hotel_dates[
        hotel_dates["_date_key"] == selected_date
    ].iloc[0]

    # Find the model probabilities for this hotel and date
    reference_data = booking_data[
        booking_data["hotel"] == selected_hotel
    ].copy()

    reference_data["_date_key"] = pd.to_datetime(
        reference_data["arrival_date"],
        errors="coerce"
    ).dt.strftime("%Y-%m-%d")

    reference_day = reference_data[
        reference_data["_date_key"] == selected_date
    ]

    if not reference_day.empty:
        baseline_cancel_pct = (
            reference_day["cancellation_probability"].mean() * 100
        )
        risk_source = "selected date's historical test sample"

    elif not reference_data.empty:
        baseline_cancel_pct = (
            reference_data["cancellation_probability"].mean() * 100
        )
        risk_source = "hotel-wide historical test sample"

    else:
        baseline_cancel_pct = 25.0
        risk_source = "illustrative fallback assumption"

    baseline_cancel_pct = float(
        np.clip(baseline_cancel_pct, 0, 100)
    )

    # --------------------------------------------------
    # 2. MANAGER INPUTS
    # --------------------------------------------------

    st.markdown("---")
    st.markdown("### 2. Adjust the business conditions")

    st.info(
        "Change these values to represent the situation you want to test. "
        "The prefilled reservation count comes from a partial historical "
        "test sample. Replace it with the hotel's complete current "
        "reservation count for a realistic operational scenario."
    )

    col1, col2 = st.columns(2)

    with col1:

        capacity = st.number_input(
            "Available room capacity",
            min_value=1,
            max_value=10000,
            value=max(1, int(selected_row["room_capacity"])),
            step=1,
            help="Enter the actual number of rooms available for this date.",
            key="whatif_capacity"
        )

        existing_bookings = st.number_input(
            "Existing reservations",
            min_value=0,
            max_value=10000,
            value=max(0, int(selected_row["total_bookings"])),
            step=1,
            help="Replace the prefilled sample count with the hotel's actual reservations.",
            key="whatif_bookings"
        )

        cancellation_pct = st.number_input(
            "Expected cancellation probability (%)",
            min_value=0.0,
            max_value=100.0,
            value=round(baseline_cancel_pct, 1),
            step=0.5,
            format="%.1f",
            help="Defaults to the average calibrated model probability for the selected historical sample. You can change it.",
            key="whatif_cancel"
        )

    with col2:

        room_rate = st.number_input(
            "Average revenue per occupied room (€)",
            min_value=0.0,
            max_value=100000.0,
            value=120.0,
            step=5.0,
            help="Illustrative business input. Replace it with the hotel's expected room revenue for the relevant stay period.",
            key="whatif_room_rate"
        )

        relocation_cost_input = st.number_input(
            "Relocation cost per guest (€)",
            min_value=0.0,
            max_value=100000.0,
            value=float(relocation_cost),
            step=10.0,
            help="Estimated cost when a guest must be accommodated elsewhere.",
            key="whatif_relocation"
        )

        risk_tolerance_pct = st.slider(
            "Maximum acceptable overcapacity risk (%)",
            min_value=0.0,
            max_value=20.0,
            value=5.0,
            step=0.5,
            help="The manager's maximum acceptable probability of arrivals exceeding room capacity.",
            key="whatif_risk"
        )

    max_additional = st.slider(
        "Maximum additional reservations to test",
        min_value=0,
        max_value=100,
        value=min(30, int(capacity)),
        step=1,
        key="whatif_max_extra"
    )

    n_simulations = 5000

    st.caption(
        f"Cancellation probability reference: {risk_source}. "
        f"Simulation count: {n_simulations:,} trials."
    )

    # --------------------------------------------------
    # 3. MONTE CARLO SIMULATION
    # --------------------------------------------------

    st.markdown("---")
    st.markdown("### 3. Simulate different booking scenarios")

    st.write(
        "HotelWise simulates possible guest arrivals. Each reservation "
        "has the same cancellation probability entered above. "
        "This is a simplified assumption for the current prototype."
    )

    if st.button(
        "Run What-If Simulation",
        type="primary",
        use_container_width=True
    ):

        # Probability that a reservation becomes an arrival
        show_probability = 1 - (cancellation_pct / 100)

        # Reproducible simulation
        rng = np.random.default_rng(42)

        # Simulate arrivals from existing reservations
        existing_arrivals = rng.binomial(
            n=int(existing_bookings),
            p=show_probability,
            size=n_simulations
        )

        # Simulate additional reservations
        if max_additional > 0:

            extra_arrivals = rng.binomial(
                n=1,
                p=show_probability,
                size=(n_simulations, max_additional)
            )

            cumulative_extra_arrivals = np.cumsum(
                extra_arrivals,
                axis=1
            )

        else:
            cumulative_extra_arrivals = np.zeros(
                (n_simulations, 0)
            )

        # Evaluate every additional-booking scenario
        results = []

        for extra in range(max_additional + 1):

            if extra == 0:
                total_arrivals = existing_arrivals.copy()
            else:
                total_arrivals = (
                    existing_arrivals
                    + cumulative_extra_arrivals[:, extra - 1]
                )

            excess_guests = np.maximum(
                total_arrivals - int(capacity),
                0
            )

            empty_rooms = np.maximum(
                int(capacity) - total_arrivals,
                0
            )

            overcapacity_risk = np.mean(
                total_arrivals > int(capacity)
            ) * 100

            expected_arrivals = np.mean(total_arrivals)

            expected_empty_rooms = np.mean(empty_rooms)

            expected_relocated = np.mean(excess_guests)

            expected_revenue = (
                np.mean(np.minimum(total_arrivals, int(capacity)))
                * room_rate
            )

            expected_relocation_cost = (
                expected_relocated * relocation_cost_input
            )

            expected_net_contribution = (
                expected_revenue - expected_relocation_cost
            )

            results.append({
                "Additional Reservations": extra,
                "Overcapacity Risk (%)": overcapacity_risk,
                "Expected Arrivals": expected_arrivals,
                "Expected Empty Rooms": expected_empty_rooms,
                "Expected Relocated Guests": expected_relocated,
                "Expected Room Revenue (€)": expected_revenue,
                "Expected Relocation Cost (€)": expected_relocation_cost,
                "Expected Net Contribution (€)": expected_net_contribution
            })

        scenario_results = pd.DataFrame(results)

        # Compare each scenario with accepting zero additional bookings
        baseline_contribution = scenario_results.iloc[0][
            "Expected Net Contribution (€)"
        ]

        scenario_results["Incremental Contribution (€)"] = (
            scenario_results["Expected Net Contribution (€)"]
            - baseline_contribution
        )

        # --------------------------------------------------
        # 4. SELECT THE BEST SCENARIO
        # --------------------------------------------------

        feasible_scenarios = scenario_results[
            scenario_results["Overcapacity Risk (%)"]
            <= risk_tolerance_pct
        ]

        if feasible_scenarios.empty:

            recommended_row = scenario_results.iloc[0]
            recommended_extra = 0
            no_feasible_scenario = True

        else:

            best_index = feasible_scenarios[
                "Expected Net Contribution (€)"
            ].idxmax()

            recommended_row = scenario_results.loc[best_index]

            recommended_extra = int(
                recommended_row["Additional Reservations"]
            )

            no_feasible_scenario = False

        # --------------------------------------------------
        # 5. RECOMMENDATION
        # --------------------------------------------------

        st.markdown("---")
        st.markdown("### 4. HotelWise Recommendation")

        metric1, metric2 = st.columns(2)
        metric3, metric4 = st.columns(2)

        metric1.metric(
            "Recommended Additional Reservations",
            f"{recommended_extra:,}"
        )

        metric2.metric(
            "Overcapacity Risk",
            f"{recommended_row['Overcapacity Risk (%)']:.2f}%"
        )

        metric3.metric(
            "Expected Net Contribution",
            f"€{recommended_row['Expected Net Contribution (€)']:,.2f}"
        )

        metric4.metric(
            "Expected Relocated Guests",
            f"{recommended_row['Expected Relocated Guests']:.2f}"
        )

        if no_feasible_scenario:

            st.error(
                "The current reservation scenario already exceeds "
                "the manager's acceptable risk limit. HotelWise does "
                "not recommend accepting additional reservations."
            )

        elif recommended_extra > 0:

            st.success(
                f"HotelWise recommends testing up to "
                f"{recommended_extra} additional reservations. "
                f"This scenario has an estimated overcapacity risk "
                f"of {recommended_row['Overcapacity Risk (%)']:.2f}%, "
                f"within the selected risk limit."
            )

        else:

            st.info(
                "HotelWise recommends accepting no additional "
                "reservations. Under the current assumptions, "
                "additional bookings do not improve the expected "
                "net contribution within the selected risk limit."
            )

        st.caption(
            "Expected net contribution is estimated room revenue "
            "minus expected guest relocation costs. It is not total "
            "hotel profit."
        )

        # --------------------------------------------------
        # 6. RISK CHART
        # --------------------------------------------------

        st.markdown("---")
        st.markdown("### How does risk change as we accept more bookings?")

        risk_chart = px.line(
            scenario_results,
            x="Additional Reservations",
            y="Overcapacity Risk (%)",
            markers=True,
            template="simple_white",
            color_discrete_sequence=["#B99A62"],
            title="Overcapacity Risk by Additional Reservations"
        )

        risk_chart.add_hline(
            y=risk_tolerance_pct,
            line_dash="dash",
            line_color="#C65D5D",
            annotation_text=f"Risk limit: {risk_tolerance_pct:.1f}%"
        )

        risk_chart.add_vline(
            x=recommended_extra,
            line_dash="dot",
            line_color="#142538",
            annotation_text="Recommended quantity"
        )

        risk_chart.update_layout(
            xaxis_title="Additional reservations",
            yaxis_title="Probability of exceeding capacity (%)",
            yaxis=dict(
                showgrid=True,
                gridcolor="#ECEFF3",
                range=[0, 100]
            ),
            xaxis=dict(showgrid=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400
        )

        st.plotly_chart(
            risk_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # 7. FINANCIAL SCENARIO CHART
        # --------------------------------------------------

        st.markdown("### Expected Financial Contribution")

        financial_chart = px.line(
            scenario_results,
            x="Additional Reservations",
            y="Expected Net Contribution (€)",
            markers=True,
            template="simple_white",
            color_discrete_sequence=["#142538"],
            title="Expected Net Contribution by Booking Scenario"
        )

        financial_chart.add_vline(
            x=recommended_extra,
            line_dash="dot",
            line_color="#B99A62",
            annotation_text="Recommended quantity"
        )

        financial_chart.update_layout(
            xaxis_title="Additional reservations",
            yaxis_title="Expected net contribution (€)",
            yaxis=dict(
                showgrid=True,
                gridcolor="#ECEFF3"
            ),
            xaxis=dict(showgrid=False),
            plot_bgcolor="white",
            paper_bgcolor="white",
            height=400
        )

        st.plotly_chart(
            financial_chart,
            use_container_width=True
        )

        # --------------------------------------------------
        # 8. SCENARIO COMPARISON TABLE
        # --------------------------------------------------

        st.markdown("### Compare All Booking Scenarios")

        st.dataframe(
            scenario_results.round(2),
            use_container_width=True,
            hide_index=True
        )

        # --------------------------------------------------
        # 9. METHODOLOGY AND LIMITATIONS
        # --------------------------------------------------

        with st.expander("How does this simulation work?"):

            st.write(
                "HotelWise runs 5,000 simulated arrival scenarios "
                "for each possible number of additional reservations."
            )

            st.write(
                "For each reservation, the simulator uses the "
                "selected cancellation probability to estimate "
                "whether the guest arrives."
            )

            st.write(
                "The overcapacity risk is the percentage of simulated "
                "scenarios in which arrivals exceed available rooms."
            )

            st.write(
                "The recommendation selects the scenario with the "
                "highest expected net contribution among scenarios "
                "that satisfy the manager's risk limit."
            )

            st.write(
                "The financial calculation counts revenue for rooms "
                "that can be accommodated, then subtracts the expected "
                "relocation cost for guests exceeding capacity."
            )

        st.warning(
            "Prototype limitations: the historical cancellation "
            "probability is based on a partial test sample. The "
            "simulation currently assumes one average cancellation "
            "probability for all existing and additional reservations. "
            "Room capacity, room revenue and relocation cost must be "
            "verified with real hotel data before operational use. "
            "The calculation does not include every hotel operating "
            "cost or multi-night booking effects."
        )


elif page == "Financial Impact":

    import plotly.express as px

    st.markdown("## Financial Impact")
    st.write(
        "Understand the potential financial value of smarter "
        "booking and overbooking decisions."
    )

    st.markdown("### Financial Scenario Results")

    # Display saved financial results
    st.dataframe(
        financial_results,
        use_container_width=True,
        hide_index=True
    )

    st.markdown("---")

    st.markdown("### Financial Assumptions")

    col1, col2 = st.columns(2)

    col1.metric(
        "Average Revenue per Booking — City Hotel",
        f"€{average_revenue['City Hotel']:,.2f}"
    )

    col2.metric(
        "Average Revenue per Booking — Resort Hotel",
        f"€{average_revenue['Resort Hotel']:,.2f}"
    )

    st.info(
        f"The simulation assumes a relocation cost of "
        f"€{relocation_cost:,.0f} per guest."
    )

    st.markdown("### How to Interpret the Results")

    st.write(
        "Additional bookings may increase revenue, but they also "
        "create a risk that more guests arrive than the hotel can accommodate."
    )

    st.write(
        "HotelWise estimates this trade-off using cancellation probabilities, "
        "room-capacity assumptions and potential relocation costs."
    )

    st.warning(
        "Important limitation: these results are illustrative scenarios "
        "based on partial test-period booking cohorts, not a validated "
        "forecast of annual hotel profit. Room capacities and relocation "
        "costs are assumptions, and the financial recommendations were "
        "calculated before the latest probability recalibration."
    )


elif page == "HotelWise AI Assistant":

    st.markdown("## HotelWise AI Assistant")
    st.write(
        "Your conversational decision-support assistant. "
        "Ask questions about cancellations, hotel capacity and financial impact."
    )

    # PREPARE REAL HOTELWISE DATA FOR THE AI
    hotel_stats = (
        booking_data.groupby("hotel")
        .agg(
            bookings=("actual_cancellation", "count"),
            actual_cancellation_rate=("actual_cancellation", "mean"),
            predicted_cancellation_rate=("cancellation_probability", "mean")
        )
        .reset_index()
    )

    hotel_stats["actual_cancellation_rate"] *= 100
    hotel_stats["predicted_cancellation_rate"] *= 100

    risk_columns = [
        "hotel", "arrival_date", "total_bookings",
        "room_capacity", "expected_arrivals", "expected_excess"
    ]

    risk_dates = (
        daily_summary
        .sort_values("expected_excess", ascending=False)
        [risk_columns]
        .head(10)
    )

    context = f"""
    HOTELWISE PROJECT DATA

    Dataset: Hotel Booking Demand, covering two hotels in Portugal,
    during 2015–2017. Results are based on a chronological test period.

    HOTEL PERFORMANCE:
    {hotel_stats.to_string(index=False)}

    MACHINE LEARNING MODEL RESULTS:
    {model_results.to_string(index=False)}

    DATES WITH THE HIGHEST EXPECTED CAPACITY PRESSURE:
    {risk_dates.to_string(index=False)}

    FINANCIAL SCENARIOS:
    {financial_results.to_string(index=False)}

    ASSUMPTIONS:
    Relocation cost: €{relocation_cost} per guest.
    Room capacities are hypothetical assumptions.
    The daily test data represents partial booking cohorts.
    Financial scenarios were calculated before the latest
    cancellation probability recalibration.

    IMPORTANT:
    Explain the difference between actual observations,
    model predictions and hypothetical assumptions.
    Never invent numbers or claim that the results prove
    performance in Moroccan hotels.
    """

    # CHAT HISTORY
    if "hotelwise_chat" not in st.session_state:
        st.session_state.hotelwise_chat = []

    if st.button("Clear conversation"):
        st.session_state.hotelwise_chat = []
        st.rerun()

    # DISPLAY PREVIOUS MESSAGES
    for message in st.session_state.hotelwise_chat:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    
st.markdown("### 💡 Suggested questions")

suggested_questions = [
    "What is our current cancellation risk?",
    "Which dates have the highest overbooking risk?",
    "Which hotel has the highest cancellation rate?",
    "How accurate is our prediction model?",
    "How can we reduce cancellation losses?",
    "What actions do you recommend for the next 7 days?"
]

cols = st.columns(2)

for i, q in enumerate(suggested_questions):
    with cols[i % 2]:
        if st.button(q, key=f"suggested_q_{i}", use_container_width=True):
            st.session_state["suggested_question"] = q

question = st.chat_input("Ask HotelWise a question...")

if not question:
    question = st.session_state.pop("suggested_question", None)


    if question:

        st.session_state.hotelwise_chat.append({
            "role": "user",
            "content": question
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("HotelWise is analysing your question..."):

                try:
                    answer = ask_hotelwise_ai(question, context)
                    st.markdown(answer)

                    st.session_state.hotelwise_chat.append({
                        "role": "assistant",
                        "content": answer
                    })

                except requests.exceptions.RequestException as error:
                    st.error(
                        "Could not connect to Ollama. "
                        "Check that Ollama is running."
                    )

                except (KeyError, ValueError) as error:
                    st.error(f"Unexpected response from the AI: {error}")

else:
    st.info(f"The {page} section is coming next.")





