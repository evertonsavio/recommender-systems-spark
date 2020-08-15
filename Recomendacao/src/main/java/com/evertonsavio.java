package com;

import org.apache.log4j.Level;
import org.apache.log4j.Logger;
import org.apache.spark.ml.recommendation.ALS;
import org.apache.spark.ml.recommendation.ALSModel;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

import javax.xml.crypto.Data;

import java.util.List;

import static org.apache.spark.sql.functions.col;

public class evertonsavio {

    public static void main(String[] args) {

        System.setProperty("hadoop.home.dir", "c:/hadoop");
        Logger.getLogger("org.apache").setLevel(Level.WARN);

        SparkSession spark = SparkSession.builder()
                .appName("VPP Chapter Views")
                .config("spark.sql.warehouse.dir","file:///c:/tmp/")
                .master("local[*]").getOrCreate();

        Dataset<Row> csvData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/VPPcourseViews.csv");

        csvData = csvData.withColumn("proportionWatched", col("proportionWatched").multiply(100));
        csvData.show();

        //csvData.groupBy("userId").pivot("courseId").sum("proportionWatched").show();

        ALS als = new ALS()
                .setMaxIter(10)
                .setRegParam(0.1)
                .setUserCol("userId")
                .setItemCol("courseId")
                .setRatingCol("proportionWatched");

        ALSModel model = als.fit(csvData);

        //Para nao NAN em novos usuarios.
        model.setColdStartStrategy("drop");

        //Dataset<Row> recomendacoesParaUsuario = model.recommendForAllUsers(5);
        //recomendacoesParaUsuario.show();

        //List<Row> userRecsList = recomendacoesParaUsuario.takeAsList(5);
        //for (Row r : userRecsList){
        //    int userId = r.getAs(0);
        //    String recs = r.getAs(1).toString();
        //    System.out.println("User " + userId + ". Nos podemos recomendar: " + recs);
        //    System.out.println("Esse usuario ja assistiu: ");
        //    csvData.filter("userId = " + userId).show();
        //}

        Dataset<Row> testData = spark.read()
                .option("header", true)
                .option("inferSchema", true)
                .csv("src/main/resources/VPPcourseViewsTest.csv");

        model.transform(testData).show();
        model.recommendForUserSubset(testData, 5).show();
    }
}
