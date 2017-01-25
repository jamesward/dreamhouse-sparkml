import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
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

  private def df(model: Model): DataFrame = {
    import spark.implicits._

    model.matrixFactorizationModel.map { case ((propertyId, userId), prediction) =>
      val rating: Int = if (favorites.contains(Favorite(propertyId, userId))) 1 else 0
      (propertyId, userId, rating, prediction)
    }.toSeq.toDF("propertyId", "userId", "rating", "prediction")
  }

  "train" should "be more accurate for lower regParams" in {
    val higherRegParam = DreamHouseRecommendations.train(favorites = favorites, regParam = 1)
    val lowerRegParam = DreamHouseRecommendations.train(favorites = favorites, regParam = 0.1)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val higherRegParamRmse = evaluator.evaluate(df(higherRegParam))
    val lowerRegParamRmse = evaluator.evaluate(df(lowerRegParam))

    lowerRegParamRmse should be < higherRegParamRmse
  }

  "train" should "be more accurate for higher ranks" in {
    val oneRank = DreamHouseRecommendations.train(favorites = favorites, rank = 1)
    val tenRank = DreamHouseRecommendations.train(favorites = favorites, rank = 10)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val oneRmse = evaluator.evaluate(df(oneRank))
    val tenRmse = evaluator.evaluate(df(tenRank))

    tenRmse should be < oneRmse
  }

  "train" should "be more accurate for more iterations" in {
    val oneIteration = DreamHouseRecommendations.train(favorites = favorites, maxIter = 1)
    val manyIterations = DreamHouseRecommendations.train(favorites = favorites, maxIter = 5)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val oneIterationRmse = evaluator.evaluate(df(oneIteration))
    val manyIterationsRmse = evaluator.evaluate(df(manyIterations))

    manyIterationsRmse should be < oneIterationRmse
  }

  override def afterAll {
    spark.stop()
  }

}
