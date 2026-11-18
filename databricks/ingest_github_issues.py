# Databricks notebook source
# MAGIC %md
# MAGIC # OpsForge | GitHub issues to Delta
# MAGIC Reads the latest Apache Airflow issue snapshots through the Unity Catalog external location.
# MAGIC The raw storage is read-only. Output tables are managed Unity Catalog Delta tables.
# MAGIC Run on a Unity Catalog-enabled Databricks compute resource with access to the catalog.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
from delta.tables import DeltaTable

CATALOG = "dbw_opsforge"
SCHEMA = "opsforge_monitoring"
SOURCE = "abfss://github-issues-raw@opsforgestorage.dfs.core.windows.net/github/apache/airflow/issues/*/latest.json"
ISSUES_TABLE = f"`{CATALOG}`.`{SCHEMA}`.`github_issues`"
COMMENTS_TABLE = f"`{CATALOG}`.`{SCHEMA}`.`github_issue_comments`"

spark.sql(f"USE CATALOG `{CATALOG}`")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{SCHEMA}`")

# COMMAND ----------

# The uploader writes one pretty-independent JSON object per file (not JSON lines).
# Explicit schema avoids inference drift as GitHub adds fields.
user_schema = T.StructType([
    T.StructField("login", T.StringType()),
    T.StructField("id", T.LongType()),
])
issue_schema = T.StructType([
    T.StructField("id", T.LongType()),
    T.StructField("number", T.LongType()),
    T.StructField("title", T.StringType()),
    T.StructField("state", T.StringType()),
    T.StructField("body", T.StringType()),
    T.StructField("created_at", T.StringType()),
    T.StructField("updated_at", T.StringType()),
    T.StructField("closed_at", T.StringType()),
    T.StructField("html_url", T.StringType()),
    T.StructField("comments", T.IntegerType()),
    T.StructField("user", user_schema),
    T.StructField("labels", T.ArrayType(T.StructType([T.StructField("name", T.StringType())]))),
    T.StructField("pull_request", T.StringType()),
])
comment_schema = T.StructType([
    T.StructField("id", T.LongType()),
    T.StructField("body", T.StringType()),
    T.StructField("created_at", T.StringType()),
    T.StructField("updated_at", T.StringType()),
    T.StructField("html_url", T.StringType()),
    T.StructField("user", user_schema),
])
snapshot_schema = T.StructType([
    T.StructField("repository", T.StringType()),
    T.StructField("source", T.StringType()),
    T.StructField("ingested_at", T.StringType()),
    T.StructField("issue", issue_schema),
    T.StructField("comments", T.ArrayType(comment_schema)),
])
raw = (spark.read.schema(snapshot_schema)
       .option("multiLine", "true")
       .json(SOURCE)
       .filter(F.col("repository") == "apache/airflow")
       .filter(F.col("issue.id").isNotNull()))
print(f"Source snapshots: {raw.count():,}")

# COMMAND ----------

issues = raw.select(
    F.col("repository"),
    F.col("issue.id").alias("issue_id"),
    F.col("issue.number").alias("issue_number"),
    F.col("issue.title").alias("title"),
    F.col("issue.state").alias("state"),
    F.col("issue.body").alias("body"),
    F.col("issue.user.login").alias("author_login"),
    F.col("issue.html_url").alias("url"),
    F.col("issue.comments").alias("reported_comment_count"),
    F.expr("transform(issue.labels, x -> x.name)").alias("labels"),
    F.to_timestamp("issue.created_at").alias("created_at"),
    F.to_timestamp("issue.updated_at").alias("updated_at"),
    F.to_timestamp("issue.closed_at").alias("closed_at"),
    F.to_timestamp("ingested_at").alias("ingested_at"),
).dropDuplicates(["repository", "issue_id"])

comments = (raw
    .select("repository", F.col("issue.id").alias("issue_id"),
            F.to_timestamp("ingested_at").alias("ingested_at"),
            F.explode_outer("comments").alias("comment"))
    .filter(F.col("comment.id").isNotNull())
    .select(
        "repository", "issue_id",
        F.col("comment.id").alias("comment_id"),
        F.col("comment.body").alias("body"),
        F.col("comment.user.login").alias("author_login"),
        F.col("comment.html_url").alias("url"),
        F.to_timestamp("comment.created_at").alias("created_at"),
        F.to_timestamp("comment.updated_at").alias("updated_at"),
        "ingested_at",
    )
    .dropDuplicates(["repository", "comment_id"]))

# COMMAND ----------

def upsert(df, table_name, keys):
    """Idempotent latest-snapshot upsert into a managed Unity Catalog Delta table."""
    if not spark.catalog.tableExists(table_name.replace("`", "")):
        (df.limit(0).write.format("delta").mode("ignore").saveAsTable(table_name))
    condition = " AND ".join(f"target.`{key}` = source.`{key}`" for key in keys)
    (DeltaTable.forName(spark, table_name).alias("target")
       .merge(df.alias("source"), condition)
       .whenMatchedUpdateAll(condition="source.ingested_at >= target.ingested_at")
       .whenNotMatchedInsertAll()
       .execute())

upsert(issues, ISSUES_TABLE, ["repository", "issue_id"])
print(f"Issues table updated: {ISSUES_TABLE}")
upsert(comments, COMMENTS_TABLE, ["repository", "comment_id"])
print(f"Comments table updated: {COMMENTS_TABLE}")

# COMMAND ----------

# GitHub comments may be deleted or removed from a snapshot. Remove records
# missing from the current issue snapshot for issues we have actually read.
# This is scoped to source issue IDs to avoid deleting data on partial ingestion.
seen_issues = raw.select(F.col("issue.id").alias("issue_id")).distinct()
current_comments = comments.select("repository", "issue_id", "comment_id")
stale = (spark.table(COMMENTS_TABLE)
    .join(seen_issues, "issue_id", "inner")
    .join(current_comments, ["repository", "issue_id", "comment_id"], "left_anti")
    .select("repository", "comment_id"))
if stale.limit(1).count():
    (DeltaTable.forName(spark, COMMENTS_TABLE).alias("target")
       .merge(stale.alias("stale"),
              "target.repository = stale.repository AND target.comment_id = stale.comment_id")
       .whenMatchedDelete()
       .execute())

# COMMAND ----------

print("Issues in Delta:", spark.table(ISSUES_TABLE).count())
print("Comments in Delta:", spark.table(COMMENTS_TABLE).count())
display(spark.table(ISSUES_TABLE).select("issue_number", "title", "state", "updated_at").orderBy(F.desc("updated_at")).limit(20))
