import tkinter as tk
from tkinter import scrolledtext, font
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import train_test_split, GridSearchCV # Keep GridSearchCV for now, can change to RandomizedSearchCV later
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.multiclass import OneVsRestClassifier
from sklearn.ensemble import RandomForestClassifier
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
import hashlib

### ==============================================================================
### Project Title: Malicious Command Prompt Detector
### Author: Ricky Rodrigues
### Student ID: TP057956
### Date: July 16, 2025
### Description: This script implements a machine learning-based application designed to
###             categorize Windows command prompts as benign, suspicious, or malicious.
###             The process involves training various machine learning models, selecting
###             the optimal performer, and integrating it into a graphical user interface
###             for real-time command analysis.
### ==============================================================================

### ==============================================================================
### --- CONFIGURATION PARAMETERS: REVIEW AND ADJUST AS NECESSARY ---
### ==============================================================================
### Specifies the filenames of the datasets used for model training.
DATA_FILES = ['cmd_huge_known_commented_updated.csv', 'd1.csv', 'd2.csv', 'd3.csv', 'd4.csv']
### Designates the column name within the dataset that contains the prompt text.
PROMPT_COLUMN = 'prompt'
### Designates the column name within the dataset that contains the labels ('benign', 'suspicious', 'malicious').
LABEL_COLUMN = 'Label'
### Defines the filename for saving the trained machine learning model.
MODEL_FILE = 'model.joblib'
### ==============================================================================
### --- END OF CONFIGURATION ---
### ==============================================================================

### --- Analysis Keywords and Associated Recommendations ---
### This dictionary defines a mapping of suspicious keywords to their corresponding explanations and recommended actions.
ANALYSIS_MAP = {
    "powershell": ("Uses PowerShell", "A powerful scripting tool. Ensure the script's source is trustworthy."),
    "invoke-expression": ("Executes Code", "Can run strings as commands, a common obfuscation technique."),
    "iex": ("Executes Code", "Alias for Invoke-Expression. Scrutinize the command's origin."),
    "downloadstring": ("Downloads Content", "May fetch and run remote scripts. Verify the URL is safe."),
    "net.webclient": ("Downloads Content", "May fetch and remote scripts. Verify the URL is safe."),
    "rmdir": ("Deletes Directories", "Can permanently delete folders and their contents. Double-check the path."),
    "/s": ("Recursive Action", "Often used with deletion commands to act on subfolders. Very high risk."),
    "/q": ("Quiet Mode", "Suppresses confirmation prompts, especially for deletion. High risk."),
    "del": ("Deletes Files", "Can permanently delete files. Double-check the target."),
    "remove-item": ("Deletes Files/Folders", "PowerShell command for deletion. Verify the target path."),
    "reg add": ("Registry Modification",
                "Adds entries to the Windows Registry. Unauthorized changes can cause instability."),
    "reg delete": ("Registry Modification", "Deletes entries from the Windows Registry. High risk of system damage."),
    "format": ("Formats Drive", "Erases all data on a drive. Extremely high risk."),
    "schtasks": ("Creates Scheduled Tasks", "Can create persistent tasks that run automatically."),
    "base64": ("Obfuscated Data", "May hide malicious commands or scripts within encoded text."),
    "net user": ("User Account Query", "Lists user accounts. Often used for reconnaissance."),
    "whoami": ("Current User Query", "Identifies the current user account. Used for reconnaissance."),
    "systeminfo": ("System Info Query", "Gathers detailed system configuration. Used for reconnaissance."),
}


### --- Machine Learning Model Training Implementation ---
def train_and_save_model():
    """
    Loads data from specified CSV files, performs necessary preprocessing, then trains and compares
    various machine learning models, including Logistic Regression, Naive Bayes, Linear SVC,
    and RandomForestClassifier.
    Hyperparameter tuning is conducted using GridSearchCV, including TF-IDF parameters.
    The model demonstrating the best performance, based on accuracy metrics, is subsequently
    saved for future deployment.

    Returns:
        sklearn.pipeline.Pipeline: The optimal trained machine learning pipeline,
                                   or None if an error precludes successful training.
    """
    print("Model not found or outdated. Initiating a new training process...")

    try:
        data_frames = [pd.read_csv(file) for file in DATA_FILES]
        df = pd.concat(data_frames, ignore_index=True)

        required_columns = [PROMPT_COLUMN, LABEL_COLUMN]
        if not all(col in df.columns for col in required_columns):
            missing_cols = [col for col in required_columns if col not in df.columns]
            print(f"Error: The dataset lacks essential columns: {missing_cols}")
            print(f"Please review the CONFIGURATION section at the beginning of this script.")
            print(f"\nAvailable columns identified in your files: {list(df.columns)}")
            return None

        df.dropna(subset=[PROMPT_COLUMN, LABEL_COLUMN], inplace=True)

        if df[LABEL_COLUMN].dtype == 'object':
            print("\nLabel column contains text values. Converting to numerical representation...")
            labels_lower = df[LABEL_COLUMN].str.lower()
            label_map = {'benign': 0, 'suspicious': 1, 'malicious': 2}
            df[LABEL_COLUMN] = labels_lower.map(label_map)

            if df[LABEL_COLUMN].isnull().any():
                print("Warning: Certain labels could not be converted and corresponding rows will be removed.")
                df.dropna(subset=[LABEL_COLUMN], inplace=True)

        df[LABEL_COLUMN] = df[LABEL_COLUMN].astype(int)

        print(f"\nSuccessfully loaded a total of {len(df)} prompts for analysis.")
        print("Label distribution (0=Benign, 1=Suspicious, 2=Malicious):\n", df[LABEL_COLUMN].value_counts())

        X = df[PROMPT_COLUMN]
        y = df[LABEL_COLUMN]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        ### --- Definition of Machine Learning Models and Hyperparameter Grids for GridSearchCV ---
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, random_state=42, solver='saga'),
            "Naive Bayes": MultinomialNB(),
            # --- UPDATED LINE: Increased max_iter for LinearSVC ---
            "Linear SVC": OneVsRestClassifier(LinearSVC(random_state=42, dual='auto', max_iter=10000)), # Increased max_iter
            "Random Forest": RandomForestClassifier(random_state=42)
        }

        # --- UPDATED SECTION: Reduced param_grids for faster training ---
        param_grids = {
            "Logistic Regression": {
                'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)], # Reduced options
                'tfidfvectorizer__max_df': [0.8, 1.0],           # Reduced options
                'logisticregression__C': [1, 10]                 # Reduced options
            },
            "Naive Bayes": {
                'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)], # Reduced options
                'tfidfvectorizer__max_df': [0.8, 1.0],           # Reduced options
                'multinomialnb__alpha': [0.1, 1.0]               # Reduced options
            },
            "Linear SVC": {
                'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)], # Reduced options
                'tfidfvectorizer__max_df': [0.8, 1.0],           # Reduced options
                'onevsrestclassifier__estimator__C': [1, 10]     # Reduced options
            },
            "Random Forest": {
                'tfidfvectorizer__ngram_range': [(1, 1), (1, 2)],       # Reduced options
                'tfidfvectorizer__max_df': [0.8, 1.0],                   # Reduced options
                'randomforestclassifier__n_estimators': [50, 100],       # Significantly reduced
                'randomforestclassifier__max_depth': [None, 20],         # Reduced options
                'randomforestclassifier__min_samples_split': [2, 5],     # Reduced options
                'randomforestclassifier__min_samples_leaf': [1, 2]       # Reduced options
            }
        }
        # --- END OF UPDATED SECTION ---


        best_model_name = None
        best_model_pipeline = None
        best_accuracy = 0.0
        class_names = ['Benign', 'Suspicious', 'Malicious']

        ### Iterating through each defined model to train, evaluate, and identify the top-performing one
        for name in models:
            print(f"\n--- Training and Evaluating: {name} ---")

            pipeline = make_pipeline(
                TfidfVectorizer(stop_words='english', max_features=15000),
                models[name]
            )

            ### Employing GridSearchCV for comprehensive hyperparameter optimization
            grid_search = GridSearchCV(pipeline, param_grids[name], cv=5, n_jobs=-1, scoring='accuracy', verbose=1)
            grid_search.fit(X_train, y_train)

            print(f"Optimal parameters for {name}: {grid_search.best_params_}")

            best_pipeline_for_current_model = grid_search.best_estimator_
            y_pred = best_pipeline_for_current_model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)

            print(f"Achieved Accuracy: {accuracy:.2%}")
            print(classification_report(y_test, y_pred, target_names=class_names))

            if accuracy > best_accuracy:
                best_accuracy = accuracy
                best_model_name = name
                best_model_pipeline = best_pipeline_for_current_model

        print(f"\n--- Optimal Model Identified: {best_model_name} with an Accuracy of: {best_accuracy:.2%} ---")

        ### --- Detailed Performance Analysis for the Selected Best Model ---
        y_pred_best = best_model_pipeline.predict(X_test)
        cm = confusion_matrix(y_test, y_pred_best)

        ### Visualization of the Confusion Matrix for the best performing model
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                            xticklabels=class_names, yticklabels=class_names)
        plt.title(f'Confusion Matrix for Optimal Model ({best_model_name})', fontsize=16)
        plt.ylabel('Actual Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        print("Displaying the confusion matrix plot for the best model...")
        plt.show()

        ### Persistence of the best model to disk
        joblib.dump(best_model_pipeline, MODEL_FILE)
        print(f"\n[✅] Optimal model ({best_model_name}) successfully saved to '{MODEL_FILE}'")

        ### Generation and display of a secure hash for the saved model file
        with open(MODEL_FILE, 'rb') as f:
            sha256 = hashlib.sha256(f.read()).hexdigest()
            print(f"[🔐] Model SHA256 Hash: {sha256}")

        return best_model_pipeline

    except Exception as e:
        print(f"\nAn unforeseen error occurred during the training process: {e}")
        return None


### --- GUI Application Implementation ---
class DetectorApp:
    """
    Implements a Tkinter-based graphical user interface for the CommandGuard AI system.
    This application enables users to submit commands for real-time classification
    (categorized as benign, suspicious, or malicious) and provides detailed
    explanations along with actionable recommendations.
    """

    def __init__(self, root, model):
        """
        Initializes the DetectorApp instance.

        Args:
            root (tk.Tk): The primary Tkinter window.
            model (sklearn.pipeline.Pipeline): The pre-trained machine learning model.
        """
        self.root = root
        self.model = model
        self.root.title("CommandGuard AI")
        self.root.geometry("1050x700")
        self.root.configure(bg="#F0F0F0")
        self.root.resizable(False, False)
        self._create_widgets()

    def _create_widgets(self):
        """Constructs and arranges all graphical user interface components."""
        ### --- Main Application Frame ---
        main_frame = tk.Frame(self.root, bg="#F0F0F0", padx=20, pady=20)
        main_frame.pack(expand=True, fill=tk.BOTH)

        ### --- Application Title ---
        title_label = tk.Label(main_frame, text="CommandGuard AI", font=("Helvetica", 24, "bold"),
                               bg="#F0F0F0", fg="#000000")
        title_label.pack(pady=(0, 25), fill=tk.X)

        ### --- Horizontal Content Frame (Separates Input from Results/Analysis) ---
        content_frame = tk.Frame(main_frame, bg="#F0F0F0")
        content_frame.pack(expand=True, fill=tk.BOTH)

        ### --- Left Panel: Command Input Section ---
        input_panel_frame = tk.Frame(content_frame, bg="#F0F0F0", padx=10, pady=0)
        input_panel_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(0, 10))

        input_label_frame = tk.LabelFrame(input_panel_frame, text="Enter Command", font=("Helvetica", 12, "bold"),
                                           fg="#555555", bg="#FFFFFF", bd=2, relief=tk.GROOVE, padx=10,
                                           pady=10)
        input_label_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 15))

        prompt_input_instruction_label = tk.Label(input_label_frame,
                                                   text="Enter your Windows command here for analysis:",
                                                   font=("Helvetica", 10), bg="#FFFFFF", fg="#444444", anchor="w")
        prompt_input_instruction_label.pack(pady=(0, 5), fill=tk.X)

        self.text_area = scrolledtext.ScrolledText(input_label_frame, height=10, font=("Consolas", 11), bg="#F5F5F5",
                                                   fg="#222222", insertbackground="#000000", relief=tk.FLAT,
                                                   borderwidth=1, padx=8, pady=8)
        self.text_area.pack(fill=tk.BOTH, expand=True)

        scan_button = tk.Button(input_panel_frame, text="Scan Command", font=("Helvetica", 13, "bold"),
                                command=self.check_prompt, bg="#000000", fg="#FFFFFF",
                                activebackground="#333333", activeforeground="#FFFFFF",
                                relief=tk.FLAT, pady=12, cursor="hand2")
        scan_button.pack(pady=(15, 0), fill=tk.X)

        ### --- Right Panel: Results and Analysis Display Sections ---
        info_panel_frame = tk.Frame(content_frame, bg="#F0F0F0", padx=10, pady=0)
        info_panel_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=(10, 0))

        ### --- Detection Outcome Display Section ---
        result_container_frame = tk.Frame(info_panel_frame, bg="#FFFFFF", bd=2, relief=tk.SOLID,
                                           highlightbackground="#CCCCCC", highlightthickness=1)
        result_container_frame.pack(fill=tk.X, pady=(0, 15), ipady=5)

        result_title_label = tk.Label(result_container_frame, text="DETECTION OUTCOME", font=("Helvetica", 12, "bold"),
                                       fg="#555555", bg="#FFFFFF", padx=10,
                                       anchor="w")
        result_title_label.pack(fill=tk.X, pady=(0, 5))

        self.predicted_label = tk.Label(result_container_frame, text="", font=("Helvetica", 16, "bold"),
                                         bg="#FFFFFF", fg="#222222", anchor="w", padx=10)
        self.predicted_label.pack(fill=tk.X, pady=(5, 5))

        self.confidence_label = tk.Label(result_container_frame, text="", font=("Helvetica", 12),
                                          bg="#FFFFFF", fg="#222222", anchor="w", padx=10)
        self.confidence_label.pack(fill=tk.X, pady=(0, 5))

        ### --- Analysis Summary Section ---
        analysis_container_frame = tk.Frame(info_panel_frame, bg="#FFFFFF", bd=2, relief=tk.SOLID,
                                             highlightbackground="#CCCCCC", highlightthickness=1)
        analysis_container_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10), ipady=5)

        analysis_title_label = tk.Label(analysis_container_frame, text="ANALYSIS SUMMARY",
                                         font=("Helvetica", 12, "bold"),
                                         fg="#555555", bg="#FFFFFF", padx=10,
                                         anchor="w")
        analysis_title_label.pack(fill=tk.X, pady=(0, 5))

        self.explanation_title = tk.Label(analysis_container_frame, text="Explanation:", font=("Helvetica", 10, "bold"),
                                           bg="#FFFFFF", fg="#444444", anchor="nw", padx=10)
        self.explanation_text_area = scrolledtext.ScrolledText(analysis_container_frame, height=5,
                                                               font=("Helvetica", 10),
                                                               bg="#F5F5F5", fg="#333333", relief=tk.FLAT,
                                                               borderwidth=0,
                                                               padx=10, pady=5, wrap=tk.WORD)
        self.explanation_text_area.config(state='disabled')

        self.recommendation_title = tk.Label(analysis_container_frame, text="Recommendation:",
                                             font=("Helvetica", 10, "bold"),
                                             bg="#FFFFFF", fg="#444444", anchor="nw", padx=10)
        self.recommendation_text_area = scrolledtext.ScrolledText(analysis_container_frame, height=5,
                                                                   font=("Helvetica", 10, "bold"),
                                                                   bg="#F5F5F5", fg="#008000", relief=tk.FLAT,
                                                                   borderwidth=0,
                                                                   padx=10, pady=5,
                                                                   wrap=tk.WORD)
        self.recommendation_text_area.config(state='disabled')

        self.explanation_text_area.pack(fill=tk.BOTH, expand=True)
        self.recommendation_text_area.pack(fill=tk.BOTH, expand=True)

        self.hide_analysis_section()

    def check_prompt(self):
        """
        Retrieves the command prompt text from the input area, utilizes the
        trained model to predict its classification, and subsequently updates
        the graphical user interface with the prediction result, a detailed
        explanation, and pertinent recommendations.
        """
        prompt_text = self.text_area.get("1.0", tk.END).strip()

        if not prompt_text:
            self.predicted_label.config(text="Please input a command for analysis.", fg="#FFA500")
            self.confidence_label.config(text="N/A", fg="#555555")
            self.hide_analysis_section()
            return

        try:
            prediction = self.model.predict([prompt_text])
            prediction_proba = self.model.predict_proba([prompt_text])
            confidence = prediction_proba.max() * 100

            predicted_class = prediction[0]

            default_exp_color = "#333333"

            if predicted_class == 2:
                result_text = "MALICIOUS"
                result_color = "#CC0000"
                rec_color_for_analysis = "#CC0000"
                self.show_analysis(prompt_text, "malicious", default_exp_color, rec_color_for_analysis)
            elif predicted_class == 1:
                result_text = "SUSPICIOUS"
                result_color = "#FFA500"
                rec_color_for_analysis = "#FFA500"
                self.show_analysis(prompt_text, "suspicious", default_exp_color, rec_color_for_analysis)
            else:
                result_text = "BENIGN"
                result_color = "#008000"
                rec_color_for_analysis = "#008000"
                self.show_benign_recommendation(default_exp_color, rec_color_for_analysis)

            self.predicted_label.config(text=f"{result_text}", fg=result_color)
            self.confidence_label.config(text=f"{confidence:.1f}%", fg=result_color)

        except Exception as e:
            self.predicted_label.config(text=f"Error", fg="#CC0000")
            self.confidence_label.config(text=f"N/A", fg="#555555")
            self.show_analysis_section()
            self._update_text_area(self.explanation_text_area, f"An error occurred during prediction: {e}", "#CC0000")
            self._update_text_area(self.recommendation_text_area, "", "#CC0000")

    def _update_text_area(self, text_widget, text_content, color):
        """Helper function to facilitate updating scrolledtext widgets."""
        text_widget.config(state='normal')
        text_widget.delete("1.0", tk.END)
        text_widget.insert(tk.END, text_content)
        text_widget.tag_configure("color_tag", foreground=color)
        text_widget.tag_add("color_tag", "1.0", tk.END)
        text_widget.config(state='disabled')

    def show_analysis(self, prompt_text, level, exp_color, rec_color):
        """
        Generates and presents comprehensive explanations and recommendations for
        prompts classified as suspicious or malicious, based on identified keywords.

        Args:
            prompt_text (str): The command prompt text that underwent analysis.
            level (str): The classification category ('suspicious' or 'malicious').
            exp_color (str): The color specification for the explanation text.
            rec_color (str): The color specification for the recommendation text.
        """
        self.show_analysis_section()
        prompt_lower = prompt_text.lower()
        found_keywords = []

        for keyword, (exp, rec) in ANALYSIS_MAP.items():
            if keyword in prompt_lower:
                if (exp, rec) not in found_keywords:
                    found_keywords.append((exp, rec))

        if found_keywords:
            explanations = "Identified suspicious elements:\n- " + "\n- ".join([f[0] for f in found_keywords])
            recommendations = "Recommendations:\n- " + "\n- ".join([f[1] for f in found_keywords])
        else:
            explanations = (f"The model detected patterns indicative of known {level} scripts, "
                            "although no specific high-risk keywords were explicitly identified in this instance.")
            recommendations = ("Exercise caution and meticulously review the command's intended purpose and origin. "
                               "Obfuscated scripts can often trigger the model's detection without overtly revealing keywords.")

        self._update_text_area(self.explanation_text_area, explanations, exp_color)
        self._update_text_area(self.recommendation_text_area, recommendations, rec_color)

    def show_benign_recommendation(self, exp_color, rec_color):
        """Displays a standard informative message and recommendation for prompts classified as benign."""
        self.show_analysis_section()
        self._update_text_area(self.explanation_text_area, "No suspicious or malicious indicators were identified.",
                               exp_color)
        self._update_text_area(self.recommendation_text_area, "The command appears safe for execution.", rec_color)

    def show_analysis_section(self):
        """Ensures that all analysis-related widgets are properly packed and visible within the GUI."""
        self.explanation_title.pack(fill=tk.X, pady=(5, 0))
        self.explanation_text_area.pack(fill=tk.BOTH, expand=True)
        self.recommendation_title.pack(fill=tk.X, pady=(10, 0))
        self.recommendation_text_area.pack(fill=tk.BOTH, expand=True)

    def hide_analysis_section(self):
        """Conceals the analysis-related widgets when they are not actively required."""
        self.explanation_title.pack_forget()
        self.explanation_text_area.pack_forget()
        self.recommendation_title.pack_forget()
        self.recommendation_text_area.pack_forget()


### --- Main Execution Block ---
if __name__ == "__main__":
    if os.path.exists(MODEL_FILE):
        print("An outdated model file was detected. Deleting it to enforce retraining with the updated algorithm.")
        os.remove(MODEL_FILE)

    model_pipeline = train_and_save_model()

    if model_pipeline:
        app_root = tk.Tk()
        app = DetectorApp(app_root, model_pipeline)
        app_root.mainloop()
    else:
        print("\nApplication could not be initiated as the model is currently unavailable.")
        try:
            error_root = tk.Tk()
            error_root.title("Error")
            error_root.geometry("450x120")
            tk.Label(error_root, text="Failed to load or train the model.\nPlease consult the console for error details.",
                     fg="red", font=("Helvetica", 12)).pack(pady=20, padx=10)
            error_root.mainloop()
        except tk.TclError:
            pass