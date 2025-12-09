import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import pickle
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="PrivacyGuard Dashboard",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #2E86AB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #2E86AB;
    }
    .stAlert {
        background-color: #d4edda;
        border-color: #c3e6cb;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load experimental results"""
    results_dir = Path('./results')
    
    with open(results_dir / 'attack_matrix.json', 'r') as f:
        attack_matrix = json.load(f)
    
    with open(results_dir / 'fairness_analysis.json', 'r') as f:
        fairness_data = json.load(f)
    
    with open(results_dir / 'tradeoff_data.json', 'r') as f:
        tradeoff_data = json.load(f)
    
    try:
        with open(results_dir / 'full_results.pkl', 'rb') as f:
            full_results = pickle.load(f)
    except:
        full_results = None
    
    return attack_matrix, fairness_data, tradeoff_data, full_results

def create_attack_matrix_heatmap(attack_matrix, selected_defenses):
    """Create interactive heatmap for attack success rates"""
    
    # Filter defenses
    filtered_matrix = {k: v for k, v in attack_matrix.items() if k in selected_defenses}
    
    if not filtered_matrix:
        # No data available
        return None
    
    defenses = list(filtered_matrix.keys())
    
    # Check if any defense has attack results
    if not any(filtered_matrix[d] for d in defenses):
        return None
    
    # Get attacks from the first defense that has results
    attacks = None
    for defense in defenses:
        if filtered_matrix[defense]:
            attacks = list(filtered_matrix[defense].keys())
            break
    
    if not attacks:
        return None
    
    # Create matrix
    z_data = []
    for attack in attacks:
        row = [filtered_matrix[defense][attack]['auc'] for defense in defenses]
        z_data.append(row)
    
    # Format labels
    attack_labels = [name.replace('_', ' ').title() for name in attacks]
    defense_labels = [name.replace('_', ' ').title() for name in defenses]
    
    fig = go.Figure(data=go.Heatmap(
        z=z_data,
        x=defense_labels,
        y=attack_labels,
        colorscale='RdYlGn_r',
        zmin=0.5,
        zmax=0.75,
        text=[[f'{val:.3f}' for val in row] for row in z_data],
        texttemplate='%{text}',
        textfont={"size": 12},
        colorbar=dict(title="Attack AUC")
    ))
    
    fig.update_layout(
        title="Membership Inference Attack Success Rates",
        xaxis_title="Defense Mechanism",
        yaxis_title="Attack Type",
        height=500,
        font=dict(size=12)
    )
    
    return fig

def create_privacy_utility_plot(tradeoff_data):
    """Create interactive privacy-utility tradeoff plot"""
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('Privacy vs. Utility', 'Attack Success vs. Privacy Budget')
    )
    
    epsilon = tradeoff_data['epsilon']
    accuracy = tradeoff_data['accuracy']
    attack_auc = tradeoff_data['avg_attack_auc']
    
    # Left plot: Accuracy vs Epsilon
    fig.add_trace(
        go.Scatter(
            x=epsilon,
            y=accuracy,
            mode='lines+markers',
            name='Accuracy',
            line=dict(color='#2E86AB', width=3),
            marker=dict(size=10)
        ),
        row=1, col=1
    )
    
    # Right plot: Attack AUC vs Epsilon
    fig.add_trace(
        go.Scatter(
            x=epsilon,
            y=attack_auc,
            mode='lines+markers',
            name='Avg Attack AUC',
            line=dict(color='#A23B72', width=3),
            marker=dict(size=10)
        ),
        row=1, col=2
    )
    
    # Add reference lines
    fig.add_hline(y=0.5, line_dash="dash", line_color="gray",
                 annotation_text="Random Guess", row=1, col=2)
    
    fig.update_xaxes(title_text="Privacy Budget (ε)", type="log", row=1, col=1)
    fig.update_xaxes(title_text="Privacy Budget (ε)", type="log", row=1, col=2)
    fig.update_yaxes(title_text="Accuracy", row=1, col=1)
    fig.update_yaxes(title_text="Attack Success (AUC)", row=1, col=2)
    
    fig.update_layout(
        height=500,
        showlegend=False,
        font=dict(size=12)
    )
    
    return fig

def create_3d_tradeoff_plot(tradeoff_data):
    """Create 3D plot for privacy-utility-fairness tradeoff"""
    
    epsilon = tradeoff_data['epsilon']
    accuracy = tradeoff_data['accuracy']
    attack_auc = tradeoff_data['avg_attack_auc']
    dp_diff = tradeoff_data['avg_dp_difference']
    
    privacy_protection = [1 - auc for auc in attack_auc]
    fairness_score = [1 - dp for dp in dp_diff]
    
    fig = go.Figure(data=[go.Scatter3d(
        x=privacy_protection,
        y=accuracy,
        z=fairness_score,
        mode='markers+text',
        marker=dict(
            size=15,
            color=epsilon,
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="ε")
        ),
        text=[f'ε={e:.1f}' for e in epsilon],
        textposition="top center"
    )])
    
    fig.update_layout(
        title="Privacy-Utility-Fairness 3D Tradeoff",
        scene=dict(
            xaxis_title="Privacy Protection",
            yaxis_title="Utility (Accuracy)",
            zaxis_title="Fairness Score"
        ),
        height=600
    )
    
    return fig

def main():
    """Main dashboard application"""
    
    # Header
    st.markdown('<div class="main-header">🔒 PrivacyGuard Enhanced Dashboard</div>',
                unsafe_allow_html=True)
    st.markdown("---")
    
    # Load data
    try:
        attack_matrix, fairness_data, tradeoff_data, full_results = load_data()
    except FileNotFoundError:
        st.error("⚠️ Results not found! Please run the experiment first: `python experiments/run_experiments.py`")
        return
    except Exception as e:
        st.error(f"⚠️ Error loading results: {e}")
        st.info("Make sure the experiment completed successfully.")
        return
    
    # Check if results are complete
    if not attack_matrix or not any(attack_matrix.values()):
        st.warning("""
        ⚠️ **Incomplete Results Detected**
        
        The experiment appears to have been interrupted. Please:
        1. Run the full experiment: `python experiments/run_experiments.py`
        2. Wait for it to complete all steps
        3. Then launch the dashboard again
        """)
        st.info("You can still explore the interface, but charts may be empty.")
    
    # Sidebar
    st.sidebar.header("⚙️ Configuration")
    
    # Defense selection
    all_defenses = list(attack_matrix.keys())
    selected_defenses = st.sidebar.multiselect(
        "Select Defense Mechanisms",
        all_defenses,
        default=all_defenses
    )
    
    # Epsilon slider for what-if analysis
    st.sidebar.markdown("---")
    st.sidebar.subheader("What-If Analysis")
    epsilon_slider = st.sidebar.slider(
        "Privacy Budget (ε)",
        min_value=0.1,
        max_value=10.0,
        value=1.0,
        step=0.1
    )
    
    # Interpolate metrics for selected epsilon
    epsilon_values = tradeoff_data['epsilon']
    if epsilon_slider in epsilon_values:
        idx = epsilon_values.index(epsilon_slider)
        estimated_accuracy = tradeoff_data['accuracy'][idx]
        estimated_attack_auc = tradeoff_data['avg_attack_auc'][idx]
        estimated_dp_diff = tradeoff_data['avg_dp_difference'][idx]
    else:
        # Linear interpolation
        estimated_accuracy = np.interp(epsilon_slider, epsilon_values,
                                      tradeoff_data['accuracy'])
        estimated_attack_auc = np.interp(epsilon_slider, epsilon_values,
                                        tradeoff_data['avg_attack_auc'])
        estimated_dp_diff = np.interp(epsilon_slider, epsilon_values,
                                      tradeoff_data['avg_dp_difference'])
    
    st.sidebar.metric("Estimated Accuracy", f"{estimated_accuracy:.3f}")
    st.sidebar.metric("Estimated Attack AUC", f"{estimated_attack_auc:.3f}")
    st.sidebar.metric("Estimated DP Difference", f"{estimated_dp_diff:.3f}")
    
    # Main content tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Overview",
        "🎯 Attack Analysis",
        "⚖️ Fairness Analysis",
        "🔄 Tradeoff Explorer"
    ])
    
    with tab1:
        st.header("Overview")
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Defense Mechanisms", len(attack_matrix))
        with col2:
            st.metric("Attack Types", len(attack_matrix[all_defenses[0]]))
        with col3:
            baseline_acc = fairness_data['baseline']['overall_accuracy']
            st.metric("Baseline Accuracy", f"{baseline_acc:.3f}")
        with col4:
            st.metric("Privacy Budgets Tested", len(tradeoff_data['epsilon']))
        
        st.markdown("---")
        
        # Summary statistics
        st.subheader("Summary Statistics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Best Privacy Protection")
            min_attack_idx = np.argmin(tradeoff_data['avg_attack_auc'])
            best_epsilon = tradeoff_data['epsilon'][min_attack_idx]
            best_acc = tradeoff_data['accuracy'][min_attack_idx]
            best_auc = tradeoff_data['avg_attack_auc'][min_attack_idx]
            
            st.info(f"""
            **ε = {best_epsilon}**
            - Accuracy: {best_acc:.3f}
            - Attack AUC: {best_auc:.3f}
            - Accuracy Loss: {(baseline_acc - best_acc):.3f}
            """)
        
        with col2:
            st.markdown("#### Recommended Setting")
            # Find "knee point" - best balance
            normalized_acc = np.array(tradeoff_data['accuracy']) / max(tradeoff_data['accuracy'])
            normalized_attack = 1 - np.array(tradeoff_data['avg_attack_auc']) / max(tradeoff_data['avg_attack_auc'])
            combined_score = normalized_acc + normalized_attack
            best_idx = np.argmax(combined_score)
            
            rec_epsilon = tradeoff_data['epsilon'][best_idx]
            rec_acc = tradeoff_data['accuracy'][best_idx]
            rec_auc = tradeoff_data['avg_attack_auc'][best_idx]
            
            st.success(f"""
            **ε = {rec_epsilon}**
            - Accuracy: {rec_acc:.3f}
            - Attack AUC: {rec_auc:.3f}
            - Good balance of privacy and utility
            """)
    
    with tab2:
        st.header("Attack Analysis")
        
        # Attack matrix heatmap
        st.subheader("Attack Resistance Matrix")
        fig_heatmap = create_attack_matrix_heatmap(attack_matrix, selected_defenses)
        if fig_heatmap is not None:
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.warning("⚠️ No attack results available yet. Please run the experiment first: `python experiments/run_experiments.py`")
        
        st.markdown("---")
        
        # Attack comparison
        st.subheader("Attack Success Comparison")
        
        # Prepare data for comparison
        attack_comparison_data = []
        for defense_name in selected_defenses:
            if defense_name in attack_matrix and attack_matrix[defense_name]:
                for attack_name, metrics in attack_matrix[defense_name].items():
                    attack_comparison_data.append({
                        'Defense': defense_name.replace('_', ' ').title(),
                        'Attack': attack_name.replace('_', ' ').title(),
                        'AUC': metrics['auc'],
                        'Accuracy': metrics['accuracy']
                })
        
        if attack_comparison_data:
            df_comparison = pd.DataFrame(attack_comparison_data)
            
            fig_comparison = px.bar(
                df_comparison,
                x='Defense',
                y='AUC',
                color='Attack',
                barmode='group',
                title='Attack Success by Defense and Attack Type'
            )
            fig_comparison.update_layout(height=500)
            st.plotly_chart(fig_comparison, use_container_width=True)
            
            # Show table
            st.subheader("Detailed Metrics")
            st.dataframe(df_comparison.pivot(index='Attack', columns='Defense', values='AUC'),
                        use_container_width=True)
        else:
            st.warning("⚠️ No attack comparison data available. Run the experiment to generate results.")
    
    with tab3:
        st.header("Fairness Analysis")
        
        # Fairness metrics comparison
        st.subheader("Fairness Metrics Across Defenses")
        
        # Extract fairness metrics
        fairness_comparison = []
        for defense_name in selected_defenses:
            if defense_name in fairness_data:
                result = fairness_data[defense_name]
                fairness_comparison.append({
                    'Defense': defense_name.replace('_', ' ').title(),
                    'Overall Accuracy': result['overall_accuracy']
                })
        
        df_fairness = pd.DataFrame(fairness_comparison)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig_fairness_acc = px.bar(
                df_fairness,
                x='Defense',
                y='Overall Accuracy',
                title='Overall Accuracy by Defense'
            )
            fig_fairness_acc.update_layout(height=400)
            st.plotly_chart(fig_fairness_acc, use_container_width=True)
        
        with col2:
            # Fairness gaps
            st.markdown("#### Key Findings")
            st.warning("""
            **Fairness-Privacy Tradeoff:**
            - Stronger privacy (lower ε) increases fairness gaps
            - Minority groups experience disproportionate utility loss
            - Important consideration for real-world deployment
            """)
        
        st.markdown("---")
        
        # Detailed fairness table
        st.subheader("Detailed Fairness Metrics")
        if full_results and 'fairness_analysis' in full_results:
            st.json(fairness_data)
    
    with tab4:
        st.header("Privacy-Utility-Fairness Tradeoff Explorer")
        
        # Privacy-utility plots
        st.subheader("2D Tradeoff Curves")
        fig_tradeoff = create_privacy_utility_plot(tradeoff_data)
        st.plotly_chart(fig_tradeoff, use_container_width=True)
        
        st.markdown("---")
        
        # 3D visualization
        st.subheader("3D Tradeoff Space")
        fig_3d = create_3d_tradeoff_plot(tradeoff_data)
        st.plotly_chart(fig_3d, use_container_width=True)
        
        st.markdown("---")
        
        # Recommendations
        st.subheader("💡 Recommendations")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### High Privacy Needs")
            st.info("""
            **ε ≤ 0.5**
            - Maximum privacy protection
            - Significant utility loss expected
            - Use for sensitive applications
            """)
        
        with col2:
            st.markdown("#### Balanced Approach")
            st.success("""
            **ε = 1.0 - 2.0**
            - Good privacy-utility balance
            - Recommended for most applications
            - Acceptable fairness tradeoffs
            """)
        
        with col3:
            st.markdown("#### High Utility Needs")
            st.warning("""
            **ε ≥ 5.0**
            - Minimal utility loss
            - Weaker privacy guarantees
            - Consider for non-sensitive data
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        PrivacyGuard Enhanced | CIS 545: Data Security & Privacy | University of Michigan - Dearborn
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
