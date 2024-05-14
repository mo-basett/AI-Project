import pandas as pd
import pickle
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

decision_tree = pickle.load(open("DecT.pkl", "rb"))
pca = pickle.load(open("PCA.pkl", "rb"))
MLP = pickle.load(open("MLPCL.pkl", "rb"))
scaler = pickle.load(open("Scaler.pkl", "rb"))

MLP_bf = np.array([0 ,1 ,4 ,5 ,6 ,7])
DT_bf=np.array([0 ,2 ,5 ,6 ,7])
answers_DT = []
answers_MLP = []
def printt():
    printview=[ ]
@app.route("/predict", methods=["POST"])
def prediction():
    json_ = request.json
    query_df = pd.DataFrame(json_)
    x = scaler.transform(query_df)
    x = pca.transform(x)
    res_dt = decision_tree.predict(x)
    answers_DT = ["B" if res == 0 else "M" for res in res_dt]
    res_MLP = MLP.predict(x)
    answers_MLP = ["B" if res == 0 else "M" for res in res_MLP]
    return jsonify({"Decision Tree": answers_DT, "MLP": answers_MLP})

if __name__ == "__main__":
    app.run()












# import pandas as pd
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# decision_tree = pickle.load(open("DecT.pkl", "rb"))
# pca = pickle.load(open("PCA.pkl", "rb"))
# MLP = pickle.load(open("MLPCL.pkl", "rb"))
# scaler = pickle.load(open("Scaler.pkl", "rb"))

# MLP_bf = np.array([0 ,1 ,2 ,3 ,6])
# DT_bf=np.array([0 ,1 ,2, 3, 6])
# best=[0 ,2 ,5 ,6 ,7]

# @app.route("/predict", methods=["POST"])
# def prediction():
#     json_ = request.json
#     query_df = pd.DataFrame(json_)
#     query_df = np.array(query_df)
#     scal = scaler.transform(query_df)
#     pca_transformed = pca.transform(scal)

#     # Subset the PCA-transformed data for each model
#     dt_input_pca = pca_transformed[:, DT_bf]
#     mlp_input_pca = pca_transformed[:, MLP_bf]

#     # Predict using Decision Tree
#     res_dt = decision_tree.predict(dt_input_pca)
#     answers_DT = ["B" if res == 0 else "M" for res in res_dt]

#     # Predict using MLP
#     answers_MLP = []
#     for i in mlp_input_pca:
#         res_mlp = MLP.predict([i])
#         res_mlp = np.round(res_mlp).astype(int)
#         if res_mlp == 0:
#             res_mlp = "B"
#         else:
#             res_mlp = "M"
#         answers_MLP.append(res_mlp)

#     return jsonify({"Decision Tree": answers_DT, "MLP": answers_MLP}) 

# if __name__ == "__main__":
#     app.run()








# import pandas as pd
# import pickle
# import numpy as np
# from flask import Flask, request, jsonify

# app = Flask(__name__)

# decision_tree = pickle.load(open("DT.pkl", "rb"))
# pca = pickle.load(open("PCAX.pkl", "rb"))
# MLP = pickle.load(open("MLP.pkl", "rb"))
# scaler = pickle.load(open("Scaler.pkl", "rb"))

# MLP_bf = np.array([0 ,1 ,4 ,5 ,6 ,7])
# # DT_bf=np.array([0 ,2 ,5 ,6 ,7])


# @app.route("/predict", methods=["POST"])
# def prediction():
#     json_ = request.json
#     query_df = pd.DataFrame(json_)
#     x = scaler.transform(query_df)
#     x = pca.transform(x)
#     answers_DT = []
#     answers_MLP = []
#     for i in x:
#         res = decision_tree.predict([i])
#         if res == 0:
#             res = "B"
#         else:
#             res = "M"
#         answers_DT.append(res)
#         res2 = MLP.predict([i])
#         res2 = np.round(res2).astype(int)
#         if res2 == 0:
#             res2 = "B"
#         else:
#             res2 = "M"
#         answers_MLP.append(res2)

#     return jsonify({"Decision Tree": answers_DT, "MLP": answers_MLP})


# if __name__ == "__main__":
#     app.run()    