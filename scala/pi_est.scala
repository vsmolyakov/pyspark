import org.apache.spark.sql.SparkSession
import scala.math.random

object pi_est {

    def inside(p:Int) : Boolean = {
        val x = random * 2 - 1
        val y = random * 2 - 1
        return x * x + y * y <= 1
    }

    def main(args: Array[String]): Unit = {
        val spark = SparkSession.builder.appName("pi_est").getOrCreate()
        val NUM_SAMPLES = 1000000
        val count = spark.sparkContext.parallelize(Range(0, NUM_SAMPLES)).filter(inside).count()
        val pi_est = 4.0 * count / NUM_SAMPLES
        println(s"Pi est = $pi_est")
    }
}