from pyspark.sql import SparkSession
from pyspark.sql.functions import to_date
from pyspark.sql.functions import hour, dayofweek, to_timestamp
from pyspark.ml.clustering import KMeans
from pyspark.ml.feature import VectorAssembler
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


spark = SparkSession.builder \
    .appName("Spatial and Temporal Clustering") \
    .getOrCreate()


df = spark.read.csv("nyc_taxi_dataset/train.csv", header=True, inferSchema=True)


df = df.withColumn("pickup_time", to_timestamp("pickup_datetime")) \
       .withColumn("hour", hour("pickup_time")) \
       .withColumn("weekday", dayofweek("pickup_time"))


df_panda = df.toPandas().head()


# EDA
# Trip Duration Distribution
# Plot
plt.figure(figsize=(10,5))
sns.histplot(df_panda["trip_duration"], bins=50, kde=True)
plt.title("Trip Duration Distribution (Filtered)")
plt.xlabel("Trip Duration (seconds)")
plt.ylabel("Frequency")
plt.show()


# Trips by Hour of Day
hourly_counts = df.groupBy("hour").count().orderBy("hour").toPandas()

plt.figure(figsize=(10,5))
sns.barplot(x="hour", y="count", data=hourly_counts, palette="Blues_d")
plt.title("Number of Trips by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.show()


# Trips by Weekday
weekday_counts = df.groupBy("weekday").count().orderBy("weekday").toPandas()

weekday_map = {1: 'Sun', 2: 'Mon', 3: 'Tue', 4: 'Wed', 5: 'Thu', 6: 'Fri', 7: 'Sat'}
weekday_counts["weekday"] = weekday_counts["weekday"].map(weekday_map)

plt.figure(figsize=(10,5))
sns.barplot(x="weekday", y="count", data=weekday_counts, palette="Greens_d")
plt.title("Number of Trips by Day of Week")
plt.xlabel("Day of Week")
plt.ylabel("Number of Trips")
plt.show()


# Trips by Passenger Count
passenger_counts = df.groupBy("passenger_count").count().orderBy("passenger_count").toPandas()

plt.figure(figsize=(8,5))
sns.barplot(x="passenger_count", y="count", data=passenger_counts, palette="Oranges_d")
plt.title("Number of Trips by Passenger Count")
plt.xlabel("Passenger Count")
plt.ylabel("Number of Trips")
plt.show()


# Pickup Locations (Spatial)
# Sample data to avoid overload
pickup_pd = df.select("pickup_latitude", "pickup_longitude").sample(fraction=0.1).toPandas()

plt.figure(figsize=(8,8))
sns.scatterplot(x="pickup_longitude", y="pickup_latitude", data=pickup_pd, alpha=0.3, s=10)
plt.title("Pickup Locations in NYC (Sampled)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()


# Dropoff Locations
dropoff_pd = df.select("dropoff_latitude", "dropoff_longitude").sample(fraction=0.1).toPandas()

plt.figure(figsize=(8,8))
sns.scatterplot(x="dropoff_longitude", y="dropoff_latitude", data=dropoff_pd, alpha=0.3, s=10, color='red')
plt.title("Dropoff Locations in NYC (Sampled)")
plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.grid(True)
plt.show()


import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import contextily as ctx
from shapely.geometry import Point

# Sample PySpark DataFrame and convert to Pandas
sample_pd = df.select("pickup_latitude", "pickup_longitude", 
                      "dropoff_latitude", "dropoff_longitude") \
              .dropna() \
              .sample(fraction=0.001) \
              .toPandas()

# Create GeoDataFrames
pickup_gdf = gpd.GeoDataFrame(
    sample_pd, 
    geometry=[Point(xy) for xy in zip(sample_pd.pickup_longitude, sample_pd.pickup_latitude)],
    crs="EPSG:4326"
)

dropoff_gdf = gpd.GeoDataFrame(
    sample_pd,
    geometry=[Point(xy) for xy in zip(sample_pd.dropoff_longitude, sample_pd.dropoff_latitude)],
    crs="EPSG:4326"
)

# Project to Web Mercator
pickup_gdf = pickup_gdf.to_crs(epsg=3857)
dropoff_gdf = dropoff_gdf.to_crs(epsg=3857)

# Plot
fig, ax = plt.subplots(figsize=(12, 12))

# Plot pickups
pickup_gdf.plot(ax=ax, markersize=3, color='blue', alpha=0.3, label='Pickup')

# Plot dropoffs
dropoff_gdf.plot(ax=ax, markersize=3, color='red', alpha=0.3, label='Dropoff')

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Final touches
ax.set_title("NYC Taxi Pickups (Blue) and Dropoffs (Red)", fontsize=14)
ax.legend(loc='upper right')
ax.set_axis_off()
plt.tight_layout()
plt.show()



# Assemble spatial feature vector
spatial_assembler = VectorAssembler(
    inputCols=["pickup_latitude", "pickup_longitude"],
    outputCol="spatial_features"
)
df_spatial = spatial_assembler.transform(df)

# KMeans model (e.g., k=3 zones)
kmeans_spatial = KMeans(featuresCol="spatial_features", predictionCol="spatial_cluster", k=10)
model_spatial = kmeans_spatial.fit(df_spatial)
df_spatial_result = model_spatial.transform(df_spatial)

# Show clustered pickup zones
df_spatial_result.select("pickup_latitude", "pickup_longitude", "spatial_cluster").show()


# Assemble temporal feature vector
temporal_assembler = VectorAssembler(
    inputCols=["hour", "weekday"],
    outputCol="temporal_features"
)
df_temporal = temporal_assembler.transform(df_spatial_result)

# KMeans model (e.g., 4 temporal periods)
kmeans_temporal = KMeans(featuresCol="temporal_features", predictionCol="temporal_cluster", k=4)
model_temporal = kmeans_temporal.fit(df_temporal)
df_final = model_temporal.transform(df_temporal)

# Show sample trips with time clusters
df_final.select("pickup_datetime", "hour", "weekday", "temporal_cluster").show()


# Combine pickup & dropoff lat/lon into one feature vector
assembler = VectorAssembler(
    inputCols=[
        "pickup_latitude", "pickup_longitude",
        "dropoff_latitude", "dropoff_longitude"
    ],
    outputCol="trip_features"
)

df_combined = assembler.transform(df)

# Apply KMeans
kmeans = KMeans(featuresCol="trip_features", predictionCol="route_cluster", k=10)
model = kmeans.fit(df_combined)
df_clustered = model.transform(df_combined)

# Show results
df_clustered.select("pickup_latitude", "pickup_longitude", "dropoff_latitude", "dropoff_longitude", "route_cluster").show()


# Convert a sample of clustered data to Pandas
cluster_sample_pd = df_clustered.select(
    "pickup_latitude", "pickup_longitude", 
    "dropoff_latitude", "dropoff_longitude", 
    "route_cluster"
).dropna().sample(fraction=0.001).toPandas()

# Create GeoDataFrame for route lines (pickup to dropoff)
from shapely.geometry import LineString
import geopandas as gpd

# Create a LineString for each route
cluster_sample_pd["geometry"] = cluster_sample_pd.apply(
    lambda row: LineString([
        (row["pickup_longitude"], row["pickup_latitude"]),
        (row["dropoff_longitude"], row["dropoff_latitude"])
    ]),
    axis=1
)

# Create GeoDataFrame with routes
route_gdf = gpd.GeoDataFrame(cluster_sample_pd, geometry="geometry", crs="EPSG:4326")
route_gdf = route_gdf.to_crs(epsg=3857)  # Convert to Web Mercator

# Plot the clustered routes
import matplotlib.pyplot as plt
import contextily as ctx
import seaborn as sns

fig, ax = plt.subplots(figsize=(14, 14))

# Choose a colormap
cmap = sns.color_palette("tab10", n_colors=10)  # 10 clusters

# Plot each cluster in a different color
for cluster_id, cluster_data in route_gdf.groupby("route_cluster"):
    cluster_data.plot(
        ax=ax, linewidth=1, alpha=0.5,
        color=cmap[cluster_id % len(cmap)],
        label=f"Cluster {cluster_id}"
    )

# Add basemap
ctx.add_basemap(ax, source=ctx.providers.CartoDB.Positron)

# Final touches
ax.set_title("NYC Taxi Routes by KMeans Clusters", fontsize=16)
ax.set_axis_off()
ax.legend()
plt.tight_layout()
plt.show()




# Convert to Pandas for plotting
temporal_pd = df_final.select("hour", "weekday", "temporal_cluster").toPandas()

# Create a pivot table
pivot = temporal_pd.pivot_table(
    index="weekday",
    columns="hour",
    values="temporal_cluster",
    aggfunc=lambda x: x.value_counts().index[0]  # Most common cluster
)

# Plot the heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(pivot, cmap="tab10", cbar_kws={'label': 'Temporal Cluster'})
plt.title("Most Frequent Temporal Cluster by Hour and Weekday")
plt.xlabel("Hour of Day")
plt.ylabel("Day of Week (1=Sun, 7=Sat)")
plt.show()





plt.figure(figsize=(12, 6))
sns.countplot(data=temporal_pd, x="hour", hue="temporal_cluster", palette="tab10")
plt.title("Distribution of Temporal Clusters by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Number of Trips")
plt.legend(title="Temporal Cluster")
plt.show()
