import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist, squareform
from skbio import DistanceMatrix
from skbio.stats.distance import mantel, pcoa


# ======================================================
# data path
# ======================================================
LEVEL = 4  # food_level

FOOD_PATH   = f"./input/food_matrix/level_{LEVEL}.tsv"          # samples x foods (+ diet label column)
ENERGY_PATH = "./input/food_matrix/energy.tsv"                 # samples x [ffq_energy_kcal]
GEO_PATH    = "./input/geo/geo_distance_matrix.csv"            # samples x samples
META_PATH   = "./input/metadata/demographics_clinical.tsv"     # has sample_id column


# ======================================================
# 1. read inputs
# ======================================================
food = pd.read_csv(FOOD_PATH, sep=",", index_col=0)
energy = pd.read_csv(ENERGY_PATH, sep=",", index_col=0)
geo = pd.read_csv(GEO_PATH, sep=",", index_col=0)
meta = pd.read_csv(META_PATH, sep="\t")


# ======================================================
# 2. align indices and basic cleaning
# - make sure food/energy share same sample index
# - remove NA energy, remove all-zero foods/rows
# ======================================================
food = food.reset_index()  # keeps diet label in a column called "index" in your original
food.index = meta["sample_id"]  # replace with the correct column name in your metadata
energy.index = food.index

valid = ~energy.isna().any(axis=1)
food = food.loc[valid]
energy = energy.loc[valid]

food = food.fillna(0)

diet_info = food[["index"]].copy()  # diet label column (e.g., "Vegan_xxx")
diet_info.index = food.index

food = food.drop(columns=["index"])

# drop food columns that are all zero
food = food.loc[:, ~(food == 0).all(axis=0)]

# drop samples that are all zero
mask = ~(food == 0).all(axis=1)
food = food.loc[mask]
energy = energy.loc[mask]
diet_info = diet_info.loc[mask]


# ======================================================
# 3. normalisation + transformation
# - divide by energy
# - MinMax scale per feature
# - arcsin(sqrt(x)) transform (you can swap to CLR if needed)
# ======================================================
food_div = food.div(energy["ffq_energy_kcal"], axis=0)

scaler = MinMaxScaler()
food_scaled = pd.DataFrame(
    scaler.fit_transform(food_div),
    index=food_div.index,
    columns=food_div.columns
)

transformed = np.arcsin(np.sqrt(food_scaled))


# ======================================================
# 4. PCA visualisation by diet group
# ======================================================
pca = PCA(n_components=2)
coords = pca.fit_transform(transformed)
var_exp = pca.explained_variance_ratio_ * 100

# diet group extracted from label prefix before "_"
diet_groups = [str(x).split("_")[0] for x in diet_info["index"]]

group_colors = {
    "Mixed diet": "skyblue",
    "Flexitarian": "darkblue",
    "Pescatarian": "green",
    "Vegetarian": "orange",
    "Vegan": "red",
}
colors = [group_colors.get(g, "gray") for g in diet_groups]

pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"], index=food.index)

plt.figure(figsize=(10, 7))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=colors, alpha=0.7, edgecolors="k")
for g, c in group_colors.items():
    plt.scatter([], [], c=c, label=g)
plt.legend(title="Diet Groups")
plt.xlabel(f"PC1: {var_exp[0]:.2f}%")
plt.ylabel(f"PC2: {var_exp[1]:.2f}%")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./output/pca_level_{LEVEL}.png", dpi=300)
plt.close()


# ======================================================
# 5. food distance matrix (Euclidean on transformed space)
# ======================================================
food_dist = pd.DataFrame(
    squareform(pdist(transformed, metric="euclidean")),
    index=food.index,
    columns=food.index
)


# ======================================================
# 6. PCoA on distance matrix
# ======================================================
pcoa_res = pcoa(food_dist)
var_explained = pcoa_res.proportion_explained * 100
pcoa_df = pcoa_res.samples.iloc[:, :2].copy()
pcoa_df.index = food.index

plt.figure(figsize=(10, 7))
plt.scatter(pcoa_df["PC1"], pcoa_df["PC2"], c=colors, alpha=0.7, edgecolors="k")
for g, c in group_colors.items():
    plt.scatter([], [], c=c, label=g)
plt.legend(title="Diet Groups")
plt.xlabel(f"PC1 ({var_explained[0]:.2f}%)")
plt.ylabel(f"PC2 ({var_explained[1]:.2f}%)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./output/pcoa_level_{LEVEL}.png", dpi=300)
plt.close()


# ======================================================
# 7. Mantel test food distance vs geographic distance
# - match on shared sample IDs
# ======================================================
shared = food_dist.index.intersection(geo.index)
food_reduced = food_dist.loc[shared, shared]
geo_reduced = geo.loc[shared, shared]

dm_food = DistanceMatrix(food_reduced.values, ids=food_reduced.index.astype(str))
dm_geo = DistanceMatrix(geo_reduced.values, ids=geo_reduced.index.astype(str))

r, p_value, n = mantel(dm_food, dm_geo, method="pearson", permutations=999)
print(f"Pearson Mantel: r={r:.3f}, p={p_value:.3f}, permutations={n}")

r, p_value, n = mantel(dm_food, dm_geo, method="spearman", permutations=999)
print(f"Spearman Mantel: r={r:.3f}, p={p_value:.3f}, permutations={n}")


# ======================================================
# 8. distance-vs-distance scatter
# ======================================================
def flatten_upper_tri(mat_df: pd.DataFrame) -> np.ndarray:
    return mat_df.values[np.triu_indices_from(mat_df.values, k=1)]

flat_food = flatten_upper_tri(food_reduced)
flat_geo = flatten_upper_tri(geo_reduced)

plt.figure(figsize=(6, 6))
sns.scatterplot(x=flat_food, y=flat_geo, s=15, alpha=0.3)
sns.regplot(x=flat_food, y=flat_geo, scatter=False, color="red")
plt.xlabel("Distances from food matrix")
plt.ylabel("Distances from geographic matrix")
plt.title("Scatterplot of Distance Matrices")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"./output/distance_correlation_level_{LEVEL}.png", dpi=300)
plt.close()
