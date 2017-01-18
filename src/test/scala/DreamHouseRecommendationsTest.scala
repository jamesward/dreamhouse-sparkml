import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class DreamHouseRecommendationsTest extends FlatSpec with BeforeAndAfterAll with Matchers {

  val favorites = Seq(
    Favorite("p1", "u1"),
    Favorite("p1", "u2"),
    Favorite("p2", "u2"),
    Favorite("p1", "u3"),
    Favorite("p2", "u3"),
    Favorite("p3", "u3")
  )

  implicit lazy val spark: SparkSession = SparkSession.builder().master("local").appName("DreamHouse Recommendations").getOrCreate()

  "train" should "work" in {
    val model = DreamHouseRecommendations.train(favorites)

    model.userFactors.collect().length should be (3)
    model.itemFactors.collect().length should be (3)
  }

  "predict" should "work" in {
    val model = DreamHouseRecommendations.train(favorites)

    val result = DreamHouseRecommendations.predict(model, "u1", 10)

    result.size should be (3)

    val sortedProperties = result.toSeq.sortBy(_._2).map(_._1).reverse

    sortedProperties(0) should be ("p1")
    sortedProperties(1) should be ("p2")
    sortedProperties(2) should be ("p3")
  }

  override def afterAll {
    spark.stop()
  }

}
