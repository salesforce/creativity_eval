from IPython.display import display_html
import pandas as pd, numpy as np

def display_side_by_side(*args):
    html_str = ''
    for df in args:
        html_str += df.to_html()
    display_html(html_str.replace('table', 'table style="display:inline"'), raw=True)

def display_results(results_ps, results_rs, results_f1s):
    dfs = []
    cols = results_ps[0].keys()
    float_cols = [col for col in cols if type(results_ps[0][col]) == np.float64]

    for name, results in [("Precision", results_ps), ("Recall", results_rs), ("F1", results_f1s)]:
        df = pd.DataFrame(results)
        styled_df = df.style.set_caption(name).format("{:.3f}", subset=float_cols)
        styled_df = styled_df.background_gradient(cmap="Blues", axis=None, low=0.0, high=1.0, subset=float_cols)
        
        dfs.append(styled_df)
    display_side_by_side(*dfs)