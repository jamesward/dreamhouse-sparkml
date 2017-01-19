import org.apache.spark.sql.SparkSession
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class DreamHouseRecommendationsTest extends FlatSpec with BeforeAndAfterAll with Matchers {

  /*
    Favorites:

       | u1 | u2 | u3 | u4 | u5 |
    -----------------------------
    p1 | ** |    |    |    |    |
    p2 | ** | ** |    |    |    |
    p3 |    | ** | ** | ** |    |
    p4 |    |    | ** | ** |    |
    p5 |    |    |    |    | ** |

   Predicted:
       | u1 | u2 | u3 | u4 | u5 |
    -----------------------------
    p1 | ** | ## |    |    |    |
    p2 | ** | ** | ## | ## |    |
    p3 |    | ** | ** | ** |    |
    p4 |    | ## | ** | ** |    |
    p5 |    |    |    |    | ** |
   */

  val favorites: Seq[Favorite] = Seq(
    Favorite("p1", "u1"),
    Favorite("p2", "u1"), Favorite("p2", "u2"),
                          Favorite("p3", "u2"), Favorite("p3", "u3"), Favorite("p3", "u4"),
                                                Favorite("p4", "u3"), Favorite("p4", "u4"),
                                                                                            Favorite("p5", "u5")
  )

  implicit lazy val spark: SparkSession = SparkSession.builder().master("local[*]").appName("DreamHouse Recommendations").getOrCreate()

  "train" should "work" in {
    val model = DreamHouseRecommendations.train(favorites)

    model.userFactors.collect().length should be (5)
    model.itemFactors.collect().length should be (5)
    model.matrixFactorizationModel.size should be (5 * 5)

    val p1u1 = model.matrixFactorizationModel("p1" -> "u1")
    val p1u2 = model.matrixFactorizationModel("p1" -> "u2")
    val p1u3 = model.matrixFactorizationModel("p1" -> "u3")

    p1u1 should be > p1u2
    p1u2 should be > p1u3
  }

  "predict" should "work" in {
    val model = DreamHouseRecommendations.train(favorites)

    val result = DreamHouseRecommendations.predict(model, "u3", 10)

    result.size should be (5)

    result("p1") should be < result("p2")
    result("p2") should be < result("p3")
    result("p2") should be < result("p4")
    result("p4") should be > result("p5")
  }

  "predict" should "work with fewer results" in {
    val model = DreamHouseRecommendations.train(favorites)

    val result = DreamHouseRecommendations.predict(model, "u3", 2)

    result.size should be (2)
    result.keys should contain ("p3")
    result.keys should contain ("p4")
  }

  override def afterAll {
    spark.stop()
  }

}
