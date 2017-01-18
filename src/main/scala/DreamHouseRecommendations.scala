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

  def train(favorites: Seq[Favorite])(implicit spark: SparkSession): Model = {
    val ratings = spark.sparkContext.makeRDD(favorites).map { favorite =>
      Rating(favorite.userId, favorite.propertyId, 1)
    }

    val (userFactors, itemFactors) = ALS.train(ratings = ratings, regParam = 0.01)

    Model(userFactors, itemFactors)
  }

  def predict(model: Model, userId: String, numResults: Int)(implicit spark: SparkSession): Map[String, Float] = {
    model.userFactors.lookup(userId).headOption.fold(Map.empty[String, Float]) { user =>

      val ratings = model.itemFactors.map { case (id, features) =>
        val rating = blas.sdot(features.length, user, 1, features, 1)
        (id, rating)
      }

      ratings.sortBy(_._2).take(numResults).toMap
    }
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

case class Model(userFactors: RDD[(String, Array[Float])], itemFactors: RDD[(String, Array[Float])])
