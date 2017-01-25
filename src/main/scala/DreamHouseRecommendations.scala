import com.github.fommil.netlib.BLAS.{getInstance => blas}
import io.circe.Decoder
import io.circe.syntax._
import org.apache.spark.ml.recommendation.ALS.Rating
import org.apache.spark.ml.recommendation.ALS
import org.apache.spark.rdd.RDD
import org.apache.spark.sql.SparkSession
import org.http4s.dsl._
import org.http4s.{HttpService, Request, Uri}
import org.http4s.client.Client
import org.http4s.client.blaze.PooledHttp1Client
import org.http4s.server.blaze.BlazeBuilder
import org.http4s.circe._
import scalaz.concurrent.Task

object DreamHouseRecommendations extends App {

  def favorites(implicit httpClient: Client): Task[Seq[Favorite]] = {
    val maybeUrl = sys.env.get("DREAMHOUSE_WEB_APP_URL").flatMap { url =>
      Uri.fromString(url + "/favorite-all").toOption
    }

    maybeUrl.map { uri =>
      httpClient.expect(Request(uri = uri))(jsonOf[Seq[Favorite]])
    } getOrElse {
      Task.fail(new Exception("The DREAMHOUSE_WEB_APP_URL env var must be set"))
    }
  }

  // regParam = How much to weigh extreme values
  // rank = Number of latent features
  // maxIter = Max iterations
  def train(favorites: Seq[Favorite], regParam: Double = 0.01, rank: Int = 5, maxIter: Int = 10)(implicit spark: SparkSession): Model = {
    val ratings = spark.sparkContext.makeRDD(favorites).map { favorite =>
      Rating(favorite.userId, favorite.propertyId, 1)
    }

    val (userFactors, itemFactors) = ALS.train(ratings = ratings, regParam = regParam, rank = rank, maxIter = maxIter)

    // this evaluates the whole matrix, right here (not in Spark) so won't scale
    val predictions = itemFactors.collect().flatMap { case (propertyId, propertyFeatures) =>
      userFactors.collect().map { case (userId, userFeatures) =>
        val prediction = blas.sdot(userFeatures.length, userFeatures, 1, propertyFeatures, 1)
        (propertyId, userId) -> prediction
      }
    }

    Model(userFactors, itemFactors, predictions.toMap)
  }

  def predict(model: Model, queryUserId: String, numResults: Int)(implicit spark: SparkSession): Map[String, Float] = {

    val propertyRatings = model.matrixFactorizationModel.collect {
      case ((propertyId, userId), rating) if userId == queryUserId => propertyId -> rating
    }

    propertyRatings.toSeq.sortBy(_._2).reverse.take(numResults).toMap
  }

  implicit val spark = SparkSession.builder().master("local[*]").appName("DreamHouse Recommendations").getOrCreate()

  implicit val httpClient = PooledHttp1Client()

  val model = train(favorites.run)

  val service = HttpService {
    case GET -> Root / userId =>
      val result = predict(model, userId, 10)
      Ok(result.asJson)
  }

  val port = sys.env.getOrElse("PORT", "8080").toInt
  val builder = BlazeBuilder.bindHttp(port, "0.0.0.0").mountService(service)
  val server = builder.run

  while (!Thread.currentThread.isInterrupted) {}

  server.shutdownNow()
  httpClient.shutdownNow()
  spark.stop()
}

case class Favorite(propertyId: String, userId: String)

object Favorite {
  implicit val decode: Decoder[Favorite] = Decoder.forProduct2("sfid", "favorite__c_user__c")(Favorite.apply)
}

case class Model(userFactors: RDD[(String, Array[Float])], itemFactors: RDD[(String, Array[Float])], matrixFactorizationModel: Map[(String, String), Float])
