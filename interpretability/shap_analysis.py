import shap
import torch

def run_shap(model, descriptor_tensor):
    model.eval()

    def model_forward(x):
        return model.regressor(x).detach().cpu().numpy()

    explainer = shap.Explainer(model_forward, descriptor_tensor)
    shap_values = explainer(descriptor_tensor)

    shap.summary_plot(shap_values, descriptor_tensor)
