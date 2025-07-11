# Databricks notebook source
# MAGIC %md
# MAGIC ## Test: import `pipelines.MedallionArchitecture`
# MAGIC * El notebook vive **dentro del mismo Repo** que tu código.
# MAGIC * No movemos el working-dir; Databricks ya pone la raíz del Repo
# MAGIC   (`.../pol_mlops_project`) en `sys.path`.
# MAGIC * Por eso basta con un import “normal”.

# COMMAND ----------
# from pipelines.MedallionArchitecture import hello_world   # 👈🏽 import directo
# hello_world()                                             # ⇒ Hello World!
# print("✅ Import OK")

import json, sys, pathlib, importlib

# ----- CONFIG ----------------------------------------------------------
LEVELS_UP_TO_FILES = 3   #  tests/pipelines/notebook  -> parents[3] = .../files
# ----------------------------------------------------------------------

# 1. Ruta de workspace del notebook (sin /Workspace al principio)
wk_path = json.loads(
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().toJson()
)["extraContext"]["notebook_path"]

# 2. Subimos hasta la carpeta ".../files"
files_dir_ws = pathlib.PurePosixPath(wk_path).parents[LEVELS_UP_TO_FILES]

# 3. Ruta real en el driver (hay que anteponer "/Workspace")
files_dir_driver = f"/Workspace{files_dir_ws}"

# 4. Añadimos a sys.path si hace falta
if files_dir_driver not in sys.path:
    sys.path.insert(0, files_dir_driver)
    print(f"🔗   Añadido a sys.path → {files_dir_driver}")

# 5. Importamos el módulo
mod = importlib.import_module("pipelines.MedallionArchitecture")
hello_world = getattr(mod, "hello_world")
hello_world()          # ⇒ debería imprimir "Hello World!"


