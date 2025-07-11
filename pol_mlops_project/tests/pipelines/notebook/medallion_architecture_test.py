# Databricks notebook source
# COMMAND ----------
# MAGIC %md
import os
from importlib import import_module

notebook_path =  '/Workspace/' + os.path.dirname(dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get())
%cd $notebook_path

%cd ../../pipelines

mod = import_module("MedallionArchitecture")
hello_world = getattr(mod, "hello_world")

hello_world()
