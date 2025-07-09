import os
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import pandas as pd
import numpy as np
import joblib
import shap
from matplotlib import pyplot as plt
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from rdkit import Chem
from features import mol2alt_sentence, MolSentence, DfVec, sentences2vec
from gensim.models import word2vec
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel
from tkinter import filedialog, messagebox
import tf_keras

tqdm.pandas()

# load ProtBERT
model_dir = "./ProtBERT"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

try:
    protbert_tokenizer = AutoTokenizer.from_pretrained(model_dir, do_lower_case=False)
    protbert_model = AutoModel.from_pretrained(model_dir).to(device)
    protbert_model.eval()
except Exception as e:
    messagebox.showerror("Model loading failed", f"ProtBERT cannot be loaded: \n{e}")
    protbert_tokenizer = None
    protbert_model = None


def predict_interaction(data):
    root = tk.Tk()
    root.withdraw()  

    # Select the XGBoost model file
    xgb_model_path = filedialog.askopenfilename(title="Select the XGBoost model file", filetypes=[("Pickle files", "*.pkl")])
    if not xgb_model_path:
        print("The XGBoost model file was not selected.")
        return None, None

    # Select the Random Forest model file
    rf_model_path = filedialog.askopenfilename(title="Select the Random Forest model file", filetypes=[("Pickle files", "*.pkl")])
    if not rf_model_path:
        print("The Random Forest model file was not selected.")
        return None, None

    # load model
    xgb_model = joblib.load(xgb_model_path)
    rf_model = joblib.load(rf_model_path)

    # predict
    xgb_pred = xgb_model.predict_proba(data)[:, 1]
    rf_pred = rf_model.predict_proba(data)[:, 1]
    final_pred_prob = 0.8 * xgb_pred + 0.2 * rf_pred
    final_pred_label = (final_pred_prob >= 0.5).astype(int)
    return final_pred_prob, final_pred_label

class PredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Drug-target interaction predictor")
        self.create_widgets()
        self.df = None
        self.file_path = None  

    def create_widgets(self):
        self.frame = tk.Frame(self.root)
        self.frame.pack(padx=10, pady=10)

        self.upload_btn = tk.Button(self.frame, text="Select the CSV file", command=self.load_file)
        self.upload_btn.grid(row=0, column=0, padx=5, pady=5)

        self.file_label = tk.Label(self.frame, text="Unselected file")  
        self.file_label.grid(row=0, column=1, columnspan=4, sticky='w')

        self.extract_all_btn = tk.Button(self.frame, text="Feature extraction", command=self.extract_all_features)
        self.extract_all_btn.grid(row=1, column=0, padx=5, pady=5)

        self.shap_btn = tk.Button(self.frame, text="SHAP", command=self.shap_feature_selection_dual_files)
        self.shap_btn.grid(row=1, column=1, padx=5, pady=5)

        self.shap_btn = tk.Button(self.frame, text="Fuzzy", command=self.fuzzy_downsample_action)
        self.shap_btn.grid(row=1, column=2, padx=5, pady=5)

        self.train_btn = tk.Button(self.frame, text="Train the model", command=self.train_model_from_gui)
        self.train_btn.grid(row=1, column=3, padx=5, pady=5)

        self.predict_btn = tk.Button(self.frame, text="Start the prediction", command=self.predict)
        self.predict_btn.grid(row=2, column=0, padx=5, pady=5)

        self.save_btn = tk.Button(self.frame, text="Save the prediction results", command=self.save_results, state=tk.DISABLED)
        self.save_btn.grid(row=2, column=1, padx=5, pady=5)

        self.tree = ttk.Treeview(self.root, columns=("Label", "Pred_Prob", "Pred_Label"), show='headings')
        self.tree.heading("Label", text="True label")
        self.tree.heading("Pred_Prob", text="Prediction probability")
        self.tree.heading("Pred_Label", text="Prediction result")
        self.tree.column("Label", anchor="center")
        self.tree.column("Pred_Prob", anchor="center")
        self.tree.column("Pred_Label", anchor="center")
        self.tree.pack(fill=tk.BOTH, expand=True, pady=5)

        self.metrics_label = tk.Label(self.root, text="The evaluation indicators will be displayed here")
        self.metrics_label.pack(pady=5)

    def load_file(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if file_path:
            try:
                self.df = pd.read_csv(file_path)
                self.file_path = file_path 
                self.file_label.config(text=f"Selected file: {file_path}")
                self.predict_btn.config(state=tk.DISABLED) 
                messagebox.showinfo("Success", "The file has been loaded successfully. Please perform feature extraction.")
            except Exception as e:
                messagebox.showerror("Error", f"File reading failed: {e}")

    def extract_all_features(self):
        if self.df is None:
            messagebox.showwarning("Warning", "Please select the CSV file first.")
            return

        try:
            # 1. mol2vec
            if "SMILES" not in self.df.columns:
                messagebox.showerror("Error", "The data does not contain SMILES columns.")
                return
            model = word2vec.Word2Vec.load("./lib/model_300dim.pkl")
            self.df["mol"] = self.df["SMILES"].progress_apply(Chem.MolFromSmiles)
            self.df["sentence"] = self.df["mol"].progress_apply(
                lambda mol: mol2alt_sentence(mol, radius=1) if mol else [])
            self.df["mol2vec"] = self.df["sentence"].progress_apply(
                lambda sentence: DfVec(sentences2vec([sentence], model, unseen="UNK")))
            mol2vec_features = pd.DataFrame(self.df["mol2vec"].apply(lambda x: x.vec.squeeze()).to_list(),
                                            columns=[f"mol2vec_{i + 1}" for i in range(300)])

            # 2. ProtBERT
            if "Target Sequence" not in self.df.columns:
                messagebox.showerror("Error", "The data does not contain Target Sequence columns.")
                return
            sequences = self.df["Target Sequence"].dropna().tolist()
            sequences = [" ".join(list(seq)) for seq in sequences]
            all_embeddings = []
            batch_size = 4
            for i in tqdm(range(0, len(sequences), batch_size), desc="Extract the sequence features of the target"):
                batch = sequences[i: i + batch_size]
                inputs = protbert_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                with torch.no_grad():
                    outputs = protbert_model(**inputs)
                cls_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(cls_embeddings)
            protbert_df = pd.DataFrame(np.vstack(all_embeddings), columns=[f"feature_{i + 1}" for i in range(1024)])

            # 3. Merge all the data
            final_df = pd.concat([self.df.drop(columns=["mol", "sentence", "mol2vec"]), mol2vec_features, protbert_df],
                                 axis=1)

            # Save data
            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if save_path:
                final_df.to_csv(save_path, index=False)
                self.df = final_df
                self.predict_btn.config(state=tk.NORMAL)
                messagebox.showinfo("Success", f"Feature extraction and saving successful! File location: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed: {e}")

    def save_mol2vec_features(self):
        if self.mol2vec_df is None:
            messagebox.showwarning("Warning", "Please perform feature extraction first")
            return

        save_path = filedialog.asksaveasfilename(title="Save the feature file as a CSV format file", defaultextension=".csv",
                                                 filetypes=[("CSV Files", "*.csv")])
        if not save_path:
            return

        try:
            self.mol2vec_df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"The features have been saved to: {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {str(e)}")

    def extract_protbert_features(self):
        if protbert_tokenizer is None or protbert_model is None:
            messagebox.showerror("Error", "The ProtBERT model was not loaded correctly.")
            return

        file_path = filedialog.askopenfilename(filetypes=[("CSV", "*.csv")])
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            if "Target Sequence" not in df.columns:
                messagebox.showerror("Error", "The data does not contain Target Sequence columns.")
                return

            sequences = df["Target Sequence"].dropna().tolist()

            def preprocess_sequence(seq):
                return " ".join(list(seq))

            sequences = [preprocess_sequence(seq) for seq in sequences]

            all_embeddings = []
            batch_size = 4

            for i in tqdm(range(0, len(sequences), batch_size), desc="Extract the sequence features of the target"):
                batch = sequences[i: i + batch_size]
                inputs = protbert_tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=512)
                input_ids = inputs["input_ids"].to(device)
                attention_mask = inputs["attention_mask"].to(device)

                with torch.no_grad():
                    outputs = protbert_model(input_ids=input_ids, attention_mask=attention_mask)

                embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                all_embeddings.append(embeddings)

            all_embeddings = np.vstack(all_embeddings)
            feature_df = pd.DataFrame(all_embeddings, columns=[f"feature_{i+1}" for i in range(all_embeddings.shape[1])])

            save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if save_path:
                feature_df.to_csv(save_path, index=False)
                messagebox.showinfo("Success", f"The features have been saved to: \n{save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Feature extraction failed: \n{e}")

    def shap_feature_selection_dual_files(self):
        try:
            # select the train set file
            train_path = filedialog.askopenfilename(title="Select the training set file", filetypes=[("CSV files", "*.csv")])
            if not train_path:
                messagebox.showwarning("Warning", "The training set file was not selected.")
                return
            train_df = pd.read_csv(train_path)

            # select the test set file
            test_path = filedialog.askopenfilename(title="Select the test set file", filetypes=[("CSV files", "*.csv")])
            if not test_path:
                messagebox.showwarning("Warning", "The test set file was not selected.")
                return
            test_df = pd.read_csv(test_path)

            
            mol2vec_cols = [f'mol2vec_{i}' for i in range(1, 301)]
            protein_cols = [f'feature_{i}' for i in range(1, 1025)]

            for col in ['Label'] + protein_cols + mol2vec_cols:
                if col not in train_df.columns or col not in test_df.columns:
                    messagebox.showerror("Error", f"The data does not contain columns: {col}")
                    return

            # load data
            X_train = train_df[protein_cols].values
            y_train = train_df['Label'].values
            X_test = test_df[protein_cols].values
            y_test = test_df['Label'].values

            # train XGBoost model
            model = XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='logloss', random_state=42)
            model.fit(X_train, y_train)

            # Calculate the SHAP value
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X_train)
            if isinstance(shap_values, list):
                shap_values = shap_values[1]

            # Visualization
            save_dir = filedialog.askdirectory(title="Select the image and result save directory")
            if not save_dir:
                messagebox.showwarning("Warning", "No save directory was selected.")
                return

            shap.summary_plot(shap_values, X_train, plot_type="bar", feature_names=protein_cols, show=False)
            plt.title("SHAP Feature Importance (Bar)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "shap_feature_importance_bar.png"), dpi=300)
            plt.close()

            shap.summary_plot(shap_values, X_train, feature_names=protein_cols, show=False)
            plt.title("SHAP Feature Summary (Beeswarm)")
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, "shap_feature_summary_beeswarm.png"), dpi=300)
            plt.close()

            # Select the top 300 features
            shap_abs = np.abs(shap_values).mean(axis=0)
            feature_importance = pd.DataFrame({
                'feature': protein_cols,
                'importance': shap_abs
            }).sort_values('importance', ascending=False)

            selected_300 = feature_importance.head(300)['feature'].tolist()
            new_feature_names = [f'feature_{i + 1}' for i in range(300)]

            # Process the training set
            selected_train_feat = train_df[selected_300].copy()
            selected_train_feat.columns = new_feature_names
            selected_train_final = pd.concat([train_df[mol2vec_cols].reset_index(drop=True),
                                              selected_train_feat.reset_index(drop=True),
                                              train_df[['Label']].reset_index(drop=True)], axis=1)

            # Process the test set
            selected_test_feat = test_df[selected_300].copy()
            selected_test_feat.columns = new_feature_names
            selected_test_final = pd.concat([test_df[mol2vec_cols].reset_index(drop=True),
                                             selected_test_feat.reset_index(drop=True),
                                             test_df[['Label']].reset_index(drop=True)], axis=1)

            # Save file
            train_save_path = os.path.join(save_dir, "mol2vec_ProtBERT_train.csv")
            test_save_path = os.path.join(save_dir, "mol2vec_ProtBERT_test.csv")
            selected_train_final.to_csv(train_save_path, index=False)
            selected_test_final.to_csv(test_save_path, index=False)

            messagebox.showinfo("Success",
                                f"SHAP feature selection is complete! \n Training set save path: {train_save_path}\nTest set save path: {test_save_path}")

        except Exception as e:
            messagebox.showerror("Error", f"An error occurred during the SHAP feature selection process: \n{e}")

    def fuzzy_downsample(self, negative_features, negative_labels, positive_features, positive_labels,
                         target_count, clear_ratio=0.5, k=5):
        from sklearn.neighbors import KNeighborsClassifier
        import numpy as np

        # Combine positive and negative samples as the training set
        X_train = np.vstack([positive_features, negative_features])
        y_train = np.concatenate([positive_labels, negative_labels])

        # Train the KNN classifier
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train, y_train)

        # Predict the probability of a negative sample being judged as a positive sample
        probs = knn.predict_proba(negative_features)[:, 1]

        # Calculate the ambiguity: The closer it is to 0.5, the more blurred it becomes
        fuzziness = 1 - 2 * np.abs(probs - 0.5)

        sorted_indices = np.argsort(fuzziness)

        clear_count = int(target_count * clear_ratio)
        fuzzy_count = target_count - clear_count

        clear_indices = sorted_indices[:clear_count]
        fuzzy_indices = sorted_indices[::-1][:fuzzy_count]

        selected_indices = np.concatenate([clear_indices, fuzzy_indices])

        return negative_features[selected_indices], negative_labels[selected_indices]

    def process_samples_with_downsampling(self, df, features_columns, label_column='Label',target_negative_count=8000, clear_ratio=0.5,k=5):
        negative_samples = df[df[label_column] == 0]
        positive_samples = df[df[label_column] == 1]

        negative_features = negative_samples[features_columns].values
        negative_labels = negative_samples[label_column].values
        positive_features = positive_samples[features_columns].values
        positive_labels = positive_samples[label_column].values

        down_neg_feats, down_neg_labels = self.fuzzy_downsample(
            negative_features, negative_labels,
            positive_features, positive_labels,
            target_count=target_negative_count,
            clear_ratio=clear_ratio,
            k=5
        )

        downsampled_negative_df = pd.DataFrame(down_neg_feats, columns=features_columns)
        downsampled_negative_df[label_column] = down_neg_labels

        final_df = pd.concat([positive_samples, downsampled_negative_df], ignore_index=True)
        return final_df

    def fuzzy_downsample_action(self):
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
        if not file_path:
            return

        try:
            df = pd.read_csv(file_path)
            features_columns = [f'mol2vec_{i}' for i in range(1, 301)] + [f'feature_{i}' for i in range(1, 301)]

            for col in features_columns + ['Label']:
                if col not in df.columns:
                    messagebox.showerror("Error", f"The data does not contain columns: {col}")
                    return

            # Perform Fuzzy downsampling
            result_df = self.process_samples_with_downsampling(df, features_columns, target_negative_count=8000,clear_ratio=0.5,k=5)

            # Save results
            output_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
            if output_path:
                result_df.to_csv(output_path, index=False)
                messagebox.showinfo("Success", f"Fuzzy downsampling has been completed and the results have been saved to:\n{output_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Error occurred during the process: \n{str(e)}")

    def train_model_from_gui(self):
        try:
            # select training set files
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if not file_path:
                messagebox.showwarning("Warning", "The training set file was not selected.")
                return

            df = pd.read_csv(file_path)

            required_cols = [f'mol2vec_{i}' for i in range(1, 301)] + [f'feature_{i}' for i in range(1, 301)] + [
                'Label']
            for col in required_cols:
                if col not in df.columns:
                    messagebox.showerror("Error", f"The data does not contain columns: {col}")
                    return

            smiles_features = df.loc[:, 'mol2vec_1':'mol2vec_300']
            protein_features = df.loc[:, 'feature_1':'feature_300']
            features = pd.concat([smiles_features, protein_features], axis=1).values
            labels = df['Label'].values

            model_save_path = 'saved_models'
            os.makedirs(model_save_path, exist_ok=True)

            xgb_model = XGBClassifier(
                n_estimators=200,
                learning_rate=0.1,
                max_depth=8,
                random_state=42,
                scale_pos_weight=len(labels[labels == 0]) / len(labels[labels == 1])
            )
            xgb_model.fit(features, labels)
            joblib.dump(xgb_model, os.path.join(model_save_path, 'xgboost_model.pkl'))

            rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=12,
                random_state=42,
                class_weight='balanced'
            )
            rf_model.fit(features, labels)
            joblib.dump(rf_model, os.path.join(model_save_path, 'random_forest_model.pkl'))

            messagebox.showinfo("Training completed", "The XGBoost and Random Forest models have been trained and saved to the 'saved_models/' folder.")

        except Exception as e:
            messagebox.showerror("Training failed", f"Error occurred during model training: \n{e}")

    def predict(self):
        try:
            # select test set files
            file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
            if not file_path:
                messagebox.showwarning("Warning", "No file was selected.")
                return

            self.df = pd.read_csv(file_path)

            feature_cols = [f"mol2vec_{i}" for i in range(1, 301)] + [f"feature_{i}" for i in range(1, 301)]
            missing_cols = [col for col in feature_cols if col not in self.df.columns]
            if missing_cols:
                messagebox.showerror("Error", f"The data does not contain columns: \n{missing_cols}")
                return

            X = self.df[feature_cols]
            y = self.df["Label"] if "Label" in self.df.columns else None

            pred_prob, pred_label = predict_interaction(X)

            self.df['Pred_Prob'] = pred_prob
            self.df['Pred_Label'] = pred_label

            self.tree.delete(*self.tree.get_children())
            for i in range(len(pred_label)):
                label = y.iloc[i] if y is not None else "-"
                self.tree.insert('', 'end', values=(label, round(pred_prob[i], 4), pred_label[i]))

            if y is not None:
                acc = accuracy_score(y, pred_label)
                auc_score = roc_auc_score(y, pred_prob)
                precision, recall, _ = precision_recall_curve(y, pred_prob)
                aupr = auc(recall, precision)
                sensitivity = sum((y == 1) & (pred_label == 1)) / sum(y == 1) if sum(y == 1) > 0 else np.nan
                specificity = sum((y == 0) & (pred_label == 0)) / sum(y == 0) if sum(y == 0) > 0 else np.nan
                metrics_text = (f"ACC: {acc:.4f}, AUC: {auc_score:.4f}, AUPR: {aupr:.4f}, "
                                f"Sensitivity: {sensitivity:.4f}, Specificity: {specificity:.4f}")
                self.metrics_label.config(text=metrics_text)
            else:
                self.metrics_label.config(text="There is no real label, so the indicators cannot be calculated")

            self.save_btn.config(state=tk.NORMAL)

        except Exception as e:
            messagebox.showerror("Prediction failed", f"Error occurred: {e}")

    def save_results(self):
        save_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
        if not save_path:
            return
        try:
            self.df.to_csv(save_path, index=False)
            messagebox.showinfo("Success", f"The prediction results have been saved to {save_path}")
        except Exception as e:
            messagebox.showerror("Error", f"Save failed: {e}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PredictorGUI(root)
    root.mainloop()
