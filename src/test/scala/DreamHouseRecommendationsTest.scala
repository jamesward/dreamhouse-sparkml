import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.scalatest.{BeforeAndAfterAll, FlatSpec, Matchers}

class DreamHouseRecommendationsTest extends FlatSpec with BeforeAndAfterAll with Matchers {

  /*
    Likes:

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
    p3 | ## | ** | ** | ** |    |
    p4 |    | ## | ** | ** |    |
    p5 |    |    |    |    | ** |
   */

  val likes: Seq[(Int, Int)] = Seq(
    (1, 1),
    (2, 1), (2, 2),
            (3, 2), (3, 3), (3, 4),
                    (4, 3), (4, 4),
                                    (5, 5)
  )

  implicit lazy val spark: SparkSession = SparkSession.builder.master("local[*]").appName("DreamHouse Recommendations").getOrCreate()

  lazy val matrix: DataFrame = {
    import spark.implicits._

    val matrix = for {
      u <- 1 to 5
      p <- 1 to 5
      rating = if (likes.contains(u -> p)) 1 else 0
      s <- Seq((u, p, rating))
    } yield s

    matrix.toDF("user", "item", "rating")
  }

  "train" should "work" in {
    val model = DreamHouseRecommendations.train(likes)

    model.userFactors.count() should be (5)
    model.itemFactors.count() should be (5)

    model.userFactors.head().getAs[Seq[Long]]("features").size should be (3)
    model.itemFactors.head().getAs[Seq[Long]]("features").size should be (3)
  }

  "predict" should "work" in {
    val model = DreamHouseRecommendations.train(likes)

    val result = DreamHouseRecommendations.predict(model, 3, 10)

    result.size should be (5)

    result(1) should be < result(2)
    result(2) should be < result(3)
    result(2) should be < result(4)
    result(4) should be > result(5)
  }


  "predict" should "work with fewer results" in {
    val model = DreamHouseRecommendations.train(likes)

    val result = DreamHouseRecommendations.predict(model, 3, 2)

    result.size should be (2)
    result.keys should contain (3)
    result.keys should contain (4)
  }

  "train" should "be more accurate for lower regParams" in {
    val higherRegParam = DreamHouseRecommendations.train(likes, 1).transform(matrix)
    val lowerRegParam = DreamHouseRecommendations.train(likes, 0.1).transform(matrix)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val higherRegParamRmse = evaluator.evaluate(higherRegParam)
    val lowerRegParamRmse = evaluator.evaluate(lowerRegParam)

    lowerRegParamRmse should be < higherRegParamRmse
  }

  "train" should "be more accurate for higher ranks" in {
    val oneRank = DreamHouseRecommendations.train(likes, rank = 1).transform(matrix)
    val fiveRank = DreamHouseRecommendations.train(likes, rank = 5).transform(matrix)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val oneRmse = evaluator.evaluate(oneRank)
    val tenRmse = evaluator.evaluate(fiveRank)

    tenRmse should be < oneRmse
  }

  "train" should "be more accurate for more iterations" in {
    val oneIteration = DreamHouseRecommendations.train(likes, maxIter = 1).transform(matrix)
    val manyIterations = DreamHouseRecommendations.train(likes, maxIter = 5).transform(matrix)

    val evaluator = new RegressionEvaluator()
      .setMetricName("rmse")
      .setLabelCol("rating")
      .setPredictionCol("prediction")

    val oneIterationRmse = evaluator.evaluate(oneIteration)
    val manyIterationsRmse = evaluator.evaluate(manyIterations)

    manyIterationsRmse should be < oneIterationRmse
  }

  override def afterAll {
    spark.stop()
  }

}
