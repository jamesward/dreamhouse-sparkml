import com.google.common.collect.{BiMap, HashBiMap}
import fs2.Task
import io.circe.Decoder
import io.circe.syntax._
import org.apache.spark.ml.recommendation.{ALS, ALSModel}
import org.apache.spark.sql.{Row, SparkSession}
import org.apache.spark.sql.functions.{col, lit}
import org.http4s.dsl._
import org.http4s.{HttpService, Request, Uri}
import org.http4s.client.Client
import org.http4s.client.blaze.PooledHttp1Client
import org.http4s.server.blaze.BlazeBuilder
import org.http4s.circe._


object DreamHouseRecommendations extends App {

  def likesTask(implicit httpClient: Client): Task[Likes] = {
    val maybeUrl = sys.env.get("DREAMHOUSE_WEB_APP_URL").flatMap { url =>
      Uri.fromString(url + "/favorite-all").fold(_ => None, Some(_))
    }

    maybeUrl.map { uri =>
      httpClient.expect(Request(uri = uri))(jsonOf[Seq[Like]]).map(Likes(_))
    } getOrElse {
      Task.fail(new Exception("The DREAMHOUSE_WEB_APP_URL env var must be set"))
    }
  }

  // regParam = How much to weigh extreme values
  // rank = Number of latent features
  // maxIter = Max iterations
  def train(likes: Seq[(Int, Int)], regParam: Double = 0.01, rank: Int = 3, maxIter: Int = 10)(implicit spark: SparkSession): ALSModel = {
    import spark.implicits._

    val ratings = likes.toDF("user", "item").withColumn("rating", lit(1))

    val als = new ALS()
      .setRegParam(regParam)
      .setRank(rank)
      .setMaxIter(maxIter)
      .setNonnegative(true)

    als.fit(ratings)
  }

  def predict(model: ALSModel, queryUser: Int, numResults: Int)(implicit spark: SparkSession): Map[Int, Float] = {
    val itemsForUser = model.recommendForAllUsers(numResults).filter(col("user") === lit(queryUser))

    itemsForUser.head().getAs[Seq[Row]]("recommendations").map { r =>
      r.getAs[Int](0) -> r.getAs[Float](1)
    }.toMap
  }

  implicit val spark = SparkSession.builder().master("local[*]").appName("DreamHouse Recommendations").getOrCreate()

  implicit val httpClient = PooledHttp1Client()

  val likes = likesTask.unsafeRun()
  val model = train(likes.likes)

  val service = HttpService {
    case GET -> Root / userId =>
      val result = predict(model, likes.userBiMap.get(userId), 10)
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

case class Likes(userBiMap: BiMap[String, Int], propertyBiMap: BiMap[String, Int], likes: Seq[(Int, Int)])

object Likes {
  def apply(likes: Seq[Like]): Likes = {
    import scala.collection.JavaConversions._
    val userBiMap: BiMap[String, Int] = HashBiMap.create(likes.map(_.userId).zipWithIndex.toMap)
    val propertyBiMap: BiMap[String, Int] = HashBiMap.create(likes.map(_.propertyId).zipWithIndex.toMap)

    val likesIntInt = likes.map { like =>
      (userBiMap.get(like.userId), propertyBiMap.get(like.propertyId))
    }

    Likes(userBiMap, propertyBiMap, likesIntInt)
  }
}

case class Like(propertyId: String, userId: String)

object Like {
  implicit val decode: Decoder[Like] = Decoder.forProduct2("sfid", "favorite__c_user__c")(Like.apply)
}
