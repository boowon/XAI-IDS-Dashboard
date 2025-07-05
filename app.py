import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import confusion_matrix, classification_report
import json
import os

# Set page config
st.set_page_config(
    page_title="ML Model XAI Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .prediction-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #1f77b4;
    }
    .actual-box {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 2px solid #2ca02c;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'selected_model' not in st.session_state:
    st.session_state.selected_model = None
if 'selected_sample' not in st.session_state:
    st.session_state.selected_sample = 0

def get_model_phase_name(model_name, model_file):
    """Determine the phase name based on model name and file"""
    is_ensemble = any(keyword in model_name.lower() for keyword in ['stacking', 'boosting', 'voting'])
    
    if model_file == 'model.pkl':
        return 'Phase 3 Model' if is_ensemble else 'Phase 1 Model'
    elif model_file == 'model_mix.pkl':
        return 'Phase 4 10-Class Model' if is_ensemble else 'Phase 2 10-Class Model'
    elif model_file == 'model_mix_2.pkl':
        return 'Phase 4 5-Class Model' if is_ensemble else 'Phase 2 5-Class Model'
    else:
        return f'{model_name} Model'

# Configuration
@st.cache_data
def load_config():
    """Load configuration and paths"""
    config = {
        'data_dir': '.\\processed_datasets',
        'model_dir': '.\\saved_artifacts\\',
        'metrics_dir': '.\\metrics',
        'feature_names': ['Timestamp', 'ID', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8'],
        'models': [
            'SVM',
            'XGBoost',
            'LightGBM',
            'MLP',
            'Random_Forest',
            'Boosting_(DT)',
            'Boosting_(LR)',
            'Stacking_(CatB+XGB+LGBM)',
            'Stacking_(MLP+XGB+LGBM)',
            'Stacking_(RF+XGB+LGBM)',
            'Stacking_(SVM+XGB+MLP)'
        ],
        'model_files': {
            'model.pkl': {
                'compatible_datasets': ['original', 'gan'],
                'dataset_mapping': {
                    'original': 'original',
                    'gan': 'gan'
                }
            },
            'model_mix.pkl': {
                'compatible_datasets': ['mixed_10'],
                'dataset_mapping': {
                    'mixed_10': 'mixed_10'
                }
            },
            'model_mix_2.pkl': {
                'compatible_datasets': ['mixed_5', 'original', 'gan'],
                'dataset_mapping': {
                    'mixed_5': 'mixed_5', 
                    'original': 'original',
                    'gan': 'gan'
                }
            }
        },
        'class_names': {
            'original': ['Normal', 'DoS', 'RPM', 'Gear', 'Fuzzy'],
            'gan': ['Normal', 'DoS', 'RPM', 'Gear', 'Fuzzy'],
            'mixed_5': ['Normal', 'DoS', 'RPM', 'Gear', 'Fuzzy'],
            'mixed_10': ['Normal', 'DoS', 'RPM', 'Gear', 'Fuzzy',
                        "GAN's Normal", "GAN's DoS", "GAN's RPM", "GAN's Gear", "GAN's Fuzzy"]
        },
        'dataset_display_names': {
            'original': 'Original Test',
            'gan': 'GAN Test',
            'mixed_5': 'Mixed 5-Class Test',
            'mixed_10': 'Mixed 10-Class Test'
        }
    }
    return config

def get_explainer(model_name, model, X_train):
    """Create LIME and SHAP explainers based on model type"""
    feature_names = ['Timestamp', 'ID', 'o1', 'o2', 'o3', 'o4', 'o5', 'o6', 'o7', 'o8']
    
    # LIME explainer
    lime_explainer = LimeTabularExplainer(
        X_train.values, 
        mode="classification", 
        feature_names=feature_names
    )
    
    # SHAP explainer based on model type
    if isinstance(model, (RandomForestClassifier, CatBoostClassifier)):
        shap_explainer = shap.TreeExplainer(
            model, 
            data=X_train, 
            model_output="probability", 
            feature_perturbation="interventional"
        )
        model_type = "tree_based"
    elif isinstance(model, (lgb.LGBMClassifier, xgb.XGBClassifier)):
        shap_explainer = shap.TreeExplainer(
            model, 
            data=X_train, 
            model_output="raw", 
            feature_perturbation="interventional"
        )
        model_type = "tree_based"
    else:
        data = shap.kmeans(X_train, 100)  # Use k-means for large datasets
        shap_explainer = shap.KernelExplainer(model.predict_proba, data)
        model_type = "kernel"
    
    return lime_explainer, shap_explainer, model_type

@st.cache_data
def load_datasets():
    """Load all datasets"""
    config = load_config()
    data_dir = config['data_dir']
    
    datasets = {}
    
    try:
        # Load original dataset
        orig_train = pd.read_csv(f'{data_dir}\\original_train.csv')
        orig_test = pd.read_csv(f'{data_dir}\\original_test.csv')
        datasets['original'] = {
            'X_train': orig_train.drop('Class', axis=1),
            'X_test': orig_test.drop('Class', axis=1),
            'y_train': orig_train['Class'],
            'y_test': orig_test['Class']
        }
        
        # Load GAN dataset
        gan_train = pd.read_csv(f'{data_dir}\\gan_train.csv')
        gan_test = pd.read_csv(f'{data_dir}\\gan_test.csv')  
        datasets['gan'] = {
            'X_train': gan_train.drop('Class', axis=1),
            'X_test': gan_test.drop('Class', axis=1),
            'y_train': gan_train['Class'],
            'y_test': gan_test['Class']
        }
        
        # Load mixed remapped dataset (10 classes)
        mix_train = pd.read_csv(f'{data_dir}\\mixed_remapped_train.csv')
        mix_test = pd.read_csv(f'{data_dir}\\mixed_remapped_test.csv')
        datasets['mixed_10'] = {
            'X_train': mix_train.drop('Class', axis=1),
            'X_test': mix_test.drop('Class', axis=1),
            'y_train': mix_train['Class'],
            'y_test': mix_test['Class']
        }
        
        # Load mixed original dataset (5 classes)
        mix2_train = pd.read_csv(f'{data_dir}\\mixed_original_train.csv') 
        mix2_test = pd.read_csv(f'{data_dir}\\mixed_original_test.csv')
        datasets['mixed_5'] = {
            'X_train': mix2_train.drop('Class', axis=1),
            'X_test': mix2_test.drop('Class', axis=1),
            'y_train': mix2_train['Class'],
            'y_test': mix2_test['Class']
        }
        
    except Exception as e:
        st.error(f"Error loading datasets: {str(e)}")
        return None
        
    return datasets

@st.cache_resource
def load_model(model_path):
    """Load a pickled model"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        return model
    except Exception as e:
        st.error(f"Error loading model from {model_path}: {str(e)}")
        return None

def load_metrics(model_name, dataset_key):
    """Load saved metrics for a model and dataset combination"""
    config = load_config()
    # Format: ModelName_DatasetKey_model_metrics.json
    metrics_filename = f"{model_name}_{dataset_key}_model_metrics.json"
    metrics_path = os.path.join(config['metrics_dir'], metrics_filename)
    
    try:
        if os.path.exists(metrics_path):
            with open(metrics_path, 'r') as f:
                metrics = json.load(f)
            return metrics
        else:
            st.warning(f"Metrics file not found: {metrics_filename}")
            return None
    except Exception as e:
        st.error(f"Error loading metrics: {str(e)}")
        return None

def get_compatible_datasets(selected_model_file, config):
    """Get compatible datasets for a selected model file"""
    if selected_model_file in config['model_files']:
        compatible_datasets = config['model_files'][selected_model_file]['compatible_datasets']
        return {config['dataset_display_names'][key]: key for key in compatible_datasets}
    return {}

def calculate_confusion_metrics(y_true, y_pred, class_names):
    """Calculate detailed confusion matrix metrics"""
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, cohen_kappa_score
    
    cm = confusion_matrix(y_true, y_pred)
    
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    kappa = cohen_kappa_score(y_true, y_pred)
    
    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, support_per_class = precision_recall_fscore_support(
        y_true, y_pred, average=None, labels=range(len(class_names))
    )
    
    # True/False Positives/Negatives for each class
    n_classes = len(class_names)
    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    tn = cm.sum() - (fp + fn + tp)
    
    # Specificity for each class
    specificity = tn / (tn + fp)
    
    metrics = {
        'confusion_matrix': cm.tolist(),
        'overall': {
            'accuracy': float(accuracy),
            'precision_weighted': float(precision),
            'recall_weighted': float(recall),
            'f1_weighted': float(f1),
            'cohen_kappa': float(kappa),
            'total_samples': int(len(y_true))
        },
        'per_class': {}
    }
    
    for i, class_name in enumerate(class_names):
        if i < len(precision_per_class):
            metrics['per_class'][class_name] = {
                'precision': float(precision_per_class[i]),
                'recall': float(recall_per_class[i]),
                'f1_score': float(f1_per_class[i]),
                'specificity': float(specificity[i]),
                'support': int(support_per_class[i]),
                'true_positives': int(tp[i]),
                'false_positives': int(fp[i]),
                'false_negatives': int(fn[i]),
                'true_negatives': int(tn[i])
            }
    
    return metrics

def create_shap_contribution_plot(feature_names, shap_values, sample_idx, actual_class, predicted_class):
    """Create SHAP feature contribution plot"""
    
    fig = go.Figure()
    
    # SHAP plot
    colors_shap = ['red' if val < 0 else 'skyblue' for val in shap_values]
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=shap_values,
            name='SHAP',
            marker_color=colors_shap,
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>SHAP Value: %{y:.4f}<extra></extra>'
        )
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        height=400,
        title_text=f"SHAP Feature Contributions for Sample {sample_idx}<br>Predicted: {predicted_class} | Actual: {actual_class}",
        title_x=0.5,
        xaxis_title="Features",
        yaxis_title="SHAP Values",
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_lime_contribution_plot(feature_names, lime_values, sample_idx, actual_class, predicted_class):
    """Create LIME feature contribution plot"""
    
    fig = go.Figure()
    
    # LIME plot  
    colors_lime = ['red' if val < 0 else 'lightgreen' for val in lime_values]
    fig.add_trace(
        go.Bar(
            x=feature_names,
            y=lime_values,
            name='LIME',
            marker_color=colors_lime,
            showlegend=False,
            hovertemplate='<b>%{x}</b><br>LIME Value: %{y:.4f}<extra></extra>'
        )
    )
    
    # Add horizontal line at y=0
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    
    # Update layout
    fig.update_layout(
        height=400,
        title_text=f"LIME Feature Contributions for Sample {sample_idx}<br>Predicted: {predicted_class} | Actual: {actual_class}",
        title_x=0.5,
        xaxis_title="Features",
        yaxis_title="LIME Values",
        showlegend=False
    )
    
    fig.update_xaxes(tickangle=45)
    
    return fig

def create_confusion_matrix_plot(y_true, y_pred, class_names):
    """Create confusion matrix plot"""
    cm = confusion_matrix(y_true, y_pred)
    
    fig = px.imshow(
        cm,
        text_auto=True,
        aspect="auto",
        color_continuous_scale='Blues',
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=class_names,
        y=class_names
    )
    
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Class",
        yaxis_title="Actual Class",
        width=600,
        height=500
    )
    
    return fig

def display_confusion_metrics(metrics, class_names):
    """Display detailed confusion matrix metrics"""
    
    # Overall metrics
    st.subheader("📊 Overall Performance Metrics")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Accuracy", f"{metrics['overall']['accuracy']:.4f}")
    with col2:
        st.metric("Precision (Weighted)", f"{metrics['overall']['precision_weighted']:.4f}")
    with col3:
        st.metric("Recall (Weighted)", f"{metrics['overall']['recall_weighted']:.4f}")
    with col4:
        st.metric("F1-Score (Weighted)", f"{metrics['overall']['f1_weighted']:.4f}")
    
    col5, col6 = st.columns(2)
    with col5:
        st.metric("Cohen's Kappa", f"{metrics['overall']['cohen_kappa']:.4f}")
    with col6:
        st.metric("Total Samples", metrics['overall']['total_samples'])
    
    # Per-class metrics table
    st.subheader("📈 Per-Class Performance Metrics")
    
    per_class_data = []
    for class_name in class_names:
        if class_name in metrics['per_class']:
            class_metrics = metrics['per_class'][class_name]
            per_class_data.append({
                'Class': class_name,
                'Precision': f"{class_metrics['precision']:.4f}",
                'Recall': f"{class_metrics['recall']:.4f}",
                'F1-Score': f"{class_metrics['f1_score']:.4f}",
                'Specificity': f"{class_metrics['specificity']:.4f}",
                'Support': class_metrics['support'],
                'TP': class_metrics['true_positives'],
                'FP': class_metrics['false_positives'],
                'FN': class_metrics['false_negatives'],
                'TN': class_metrics['true_negatives']
            })
    
    if per_class_data:
        df_metrics = pd.DataFrame(per_class_data)
        st.dataframe(df_metrics, use_container_width=True)
    
    # Confusion Matrix
    st.subheader("🎯 Confusion Matrix Details")
    cm_array = np.array(metrics['confusion_matrix'])
    
    # Create a more detailed confusion matrix display
    cm_df = pd.DataFrame(cm_array, index=class_names, columns=class_names)
    cm_df.index.name = 'Actual'
    cm_df.columns.name = 'Predicted'
    
    st.dataframe(cm_df, use_container_width=True)

def main():
    st.title("🔍 ML Model XAI Analysis Dashboard")
    st.markdown("---")
    
    # Load configuration and data
    config = load_config()
    datasets = load_datasets()
    
    if datasets is None:
        st.error("Failed to load datasets. Please check your data directory path.")
        return
    
    # Sidebar for model selection
    st.sidebar.header("Model Selection")
    
    # Model selection from config list
    selected_model = st.sidebar.selectbox("Select Model", config['models'])
    
    # Model file selection
    model_files = list(config['model_files'].keys())
    selected_model_file = st.sidebar.selectbox("Select Model Version", model_files)
    
    # Get the phase-based model name
    model_phase_name = get_model_phase_name(selected_model, selected_model_file)
    
    # Get compatible datasets for the selected model file
    compatible_datasets = get_compatible_datasets(selected_model_file, config)
    
    if not compatible_datasets:
        st.error(f"No compatible datasets found for {selected_model_file}")
        return
    
    # Dataset selection (filtered by compatibility)
    selected_dataset = st.sidebar.selectbox("Select Test Dataset", list(compatible_datasets.keys()))
    dataset_key = compatible_datasets[selected_dataset]
    
    # Sample selection
    if dataset_key in datasets:
        max_samples = len(datasets[dataset_key]['X_test']) - 1
        selected_sample = st.sidebar.number_input(
            "Sample Index", 
            min_value=0, 
            max_value=max_samples, 
            value=0,
            help=f"Select a sample from 0 to {max_samples}"
        )
        
        # Main content area
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header(f"Model: {model_phase_name}")
            st.info(f"Testing on: {selected_dataset}")
            
            # Load model
            model_path = f"{config['model_dir']}\\{selected_model}\\{selected_model_file}"
            model = load_model(model_path)
            
            if model is not None:
                # Get test data
                X_test = datasets[dataset_key]['X_test']
                y_test = datasets[dataset_key]['y_test']
                X_train = datasets[dataset_key]['X_train']
                
                # Get class names for current dataset
                class_names = config['class_names'][dataset_key]
                
                # Make prediction for selected sample
                sample_data = X_test.iloc[selected_sample:selected_sample+1]
                actual_class_idx = y_test.iloc[selected_sample]
                actual_class_name = class_names[actual_class_idx] if actual_class_idx < len(class_names) else f"Class {actual_class_idx}"
               
                if selected_model == 'LSTM':
                    pred = model.predict(sample_data.values.reshape(1, -1, sample_data.shape[1]))
                    prediction_idx = np.argmax(pred, axis=1)[0]
                elif selected_model in ['CatBoost']:
                    prediction_idx = model.predict(sample_data)[0][0]
                else:
                    prediction_idx = model.predict(sample_data)[0]
                

                print(f"Prediction Index: {prediction_idx}, Actual Class Index: {actual_class_idx}")

                predicted_class_name = class_names[prediction_idx] if prediction_idx < len(class_names) else f"Class {prediction_idx}"
                prediction_proba = model.predict_proba(sample_data)[0]
                
                # Display predictions
                col1_1, col1_2 = st.columns(2)
                
                with col1_1:
                    st.markdown(f"""
                    <div class="prediction-box">
                        <h4>🔮 Predicted Class</h4>
                        <h2>{predicted_class_name}</h2>
                        <p>Index: {prediction_idx}</p>
                        <p>Confidence: {max(prediction_proba):.3f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col1_2:
                    st.markdown(f"""
                    <div class="actual-box">
                        <h4>✅ Actual Class</h4>
                        <h2>{actual_class_name}</h2>
                        <p>Index: {actual_class_idx}</p>
                        <p>Match: {'✓' if prediction_idx == actual_class_idx else '✗'}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Prediction probabilities
                st.subheader("Prediction Probabilities")
                prob_df = pd.DataFrame({
                    'Class': [class_names[i] if i < len(class_names) else f"Class {i}" for i in range(len(prediction_proba))],
                    'Class_Index': range(len(prediction_proba)),
                    'Probability': prediction_proba
                })
                
                fig_prob = px.bar(
                    prob_df, 
                    x='Class', 
                    y='Probability',
                    title="Class Prediction Probabilities",
                    hover_data=['Class_Index']
                )
                fig_prob.update_xaxes(tickangle=45)
                st.plotly_chart(fig_prob, use_container_width=True)
                
                # Feature contributions
                if st.button("🔍 Analyze Feature Contributions", type="primary"):
                    with st.spinner("Calculating feature contributions..."):
                        # Get explainers
                        lime_explainer, shap_explainer, model_type = get_explainer(
                            selected_model, model, X_train
                        )
                        
                        # Calculate SHAP values
                        shap_values = shap_explainer.shap_values(sample_data.iloc[0])
                        
                        # Handle different SHAP output formats
                        if isinstance(shap_values, list):
                            # Multi-class case - take values for predicted class
                            shap_vals = shap_values[prediction_idx]
                        else:
                            # Binary case or single output - use as is
                            if len(shap_values.shape) > 1:
                                shap_vals = shap_values[:, prediction_idx] if shap_values.shape[1] > prediction_idx else shap_values[:, 0]
                            else:
                                shap_vals = shap_values
                        
                        # Calculate LIME values
                        exp = lime_explainer.explain_instance(
                            sample_data.values[0], 
                            model.predict_proba, 
                            labels=[prediction_idx]
                        )
                        
                        lime_vals = np.zeros(len(config['feature_names']))
                        for idx, val in exp.as_map()[prediction_idx]:
                            lime_vals[idx] = val
                        
                        # Create and display separate plots
                        st.markdown("---")
                        st.subheader("🔍 Explainability Analysis")
                        
                        # SHAP plot
                        fig_shap = create_shap_contribution_plot(
                            config['feature_names'], 
                            shap_vals, 
                            selected_sample,
                            actual_class_name,
                            predicted_class_name
                        )
                        st.plotly_chart(fig_shap, use_container_width=True)
                        
                        # Add some spacing
                        st.markdown("<br>", unsafe_allow_html=True)
                        
                        # LIME plot
                        fig_lime = create_lime_contribution_plot(
                            config['feature_names'], 
                            lime_vals, 
                            selected_sample,
                            actual_class_name,
                            predicted_class_name
                        )
                        st.plotly_chart(fig_lime, use_container_width=True)
                
                # Calculate and display confusion matrix metrics
                if st.button("🎯 Calculate Confusion Matrix", type="primary"):
                    if model is not None:
                        with st.spinner("Calculating confusion matrix and metrics..."):
                            X_test = datasets[dataset_key]['X_test']
                            y_test = datasets[dataset_key]['y_test']
                            class_names = config['class_names'][dataset_key]
                            
                            # Get predictions for all test samples
                            y_pred = model.predict(X_test)
                            
                            # Calculate detailed metrics
                            confusion_metrics = calculate_confusion_metrics(y_test, y_pred, class_names)
                            
                            # Display confusion matrix plot
                            fig_cm = create_confusion_matrix_plot(y_test, y_pred, class_names)
                            st.plotly_chart(fig_cm, use_container_width=True)
                            
                            # Display detailed metrics
                            display_confusion_metrics(confusion_metrics, class_names)
        with col2:
            st.header("📊 Model Performance")
            
            # Load and display metrics
            if st.button("📈 Show Saved Metrics", type="secondary"):
                metrics = load_metrics(selected_model, dataset_key)
                
                if metrics:
                    st.subheader("Saved Performance Metrics")
                    
                    # Basic performance metrics
                    col2_1, col2_2 = st.columns(2)
                    with col2_1:
                        st.metric("Accuracy", f"{metrics.get('accuracy', 0):.3f}")
                        st.metric("Precision", f"{metrics.get('precision', 0):.3f}")
                    with col2_2:
                        st.metric("Recall", f"{metrics.get('recall', 0):.3f}")
                        st.metric("F1-Score", f"{metrics.get('f1_score', 0):.3f}")
                    
                    # XAI Quality metrics
                    st.subheader("XAI Quality Metrics")
                    col2_3, col2_4 = st.columns(2)
                    with col2_3:
                        st.metric("SHAP Fidelity", f"{metrics.get('shap_fidelity', 0):.3f}")
                        st.metric("SHAP Sparsity", f"{metrics.get('shap_sparsity', 0):.3f}")
                        st.metric("SHAP Stability", f"{metrics.get('shap_stability', 0):.3f}")
                    with col2_4:
                        st.metric("LIME Fidelity", f"{metrics.get('lime_fidelity', 0):.3f}")
                        st.metric("LIME Sparsity", f"{metrics.get('lime_sparsity', 0):.3f}")
                        st.metric("LIME Stability", f"{metrics.get('lime_stability', 0):.3f}")
                else:
                    st.info("No saved metrics found for this model-dataset combination.")
                
    # Information section
    st.markdown("---")
    st.markdown(f"""
    ### 📋 Model-Dataset Compatibility
    
    **Current Selection:**
    - **Model:** {model_phase_name}
    - **Compatible Datasets:** {', '.join(compatible_datasets.keys())}
    - **Selected Dataset:** {selected_dataset}
    """)

if __name__ == "__main__":
    main()