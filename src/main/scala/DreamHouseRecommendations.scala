import org.apache.commons.httpclient.HttpClient
import org.apache.commons.httpclient.methods.GetMethod
import org.apache.spark.mllib.recommendation.{ALS, MatrixFactorizationModel, Rating}
import org.apache.spark.{SparkConf, SparkContext}
import org.http4s._
import org.http4s.dsl._
import org.http4s.headers.`Content-Type`
import org.http4s.server.blaze._
import org.json4s.JsonAST.{JArray, JField, JObject, JString}
import org.json4s.JsonDSL._
import org.json4s.jackson.JsonMethods._

object DreamHouseRecommendations extends App {

  def favorites: Seq[Favorite] = {

    val httpClient = new HttpClient()

    val baseUrl = sys.env("DREAMHOUSE_WEB_APP_URL")

    val getFavorites = new GetMethod(baseUrl + "/favorite-all")

    httpClient.executeMethod(getFavorites)

    val json = parse(getFavorites.getResponseBodyAsStream)

    for {
      JArray(favorites) <- json
      JObject(favorite) <- favorites
      JField("sfid", JString(propertyId)) <- favorite
      JField("favorite__c_user__c", JString(userId)) <- favorite
    } yield Favorite(propertyId, userId)
  }

  private def seqToMap(s: Seq[String]): Map[String, Int] = {
    s.distinct.zipWithIndex.toMap
  }

  def train(sc: SparkContext, favorites: Seq[Favorite]): Model = {

    val userIdMap = seqToMap(favorites.map(_.userId))
    val propertyIdMap = seqToMap(favorites.map(_.propertyId))

    val ratings = favorites.map { favorite =>
      Rating(userIdMap(favorite.userId), propertyIdMap(favorite.propertyId), 1)
    }

    val rdd = sc.parallelize(ratings)

    val matrixFactorizationModel = ALS.train(rdd, 10, 10)

    Model(matrixFactorizationModel, userIdMap, propertyIdMap)
  }

  def predict(sc: SparkContext, model: Model, userId: String, numResults: Int): Map[String, Double] = {
    val userIdInt = model.userIdMap(userId)

    val ratings = model.model.recommendProducts(userIdInt, numResults)

    ratings.map { rating =>
      model.propertyIdMap.map(_.swap).apply(rating.product) -> rating.rating
    }.toMap
  }

  val conf = new SparkConf(false).setMaster("local").setAppName("dreamhouse")
  val sc = new SparkContext(conf)

  val model = train(sc, favorites)

  val service = HttpService {
    case GET -> Root / userId =>
      val result = predict(sc, model, userId, 10)
      val json = compact(render(result))
      Ok(json).withContentType(Some(`Content-Type`(MediaType.`application/json`)))
  }

  val port = sys.env.getOrElse("PORT", "8080").toInt
  val builder = BlazeBuilder.bindLocal(port).mountService(service)
  val server = builder.run

  while (!Thread.currentThread.isInterrupted) {}

  server.shutdownNow()
  sc.stop()
}

case class Favorite(propertyId: String, userId: String)

case class Model(model: MatrixFactorizationModel, userIdMap: Map[String, Int], propertyIdMap: Map[String, Int])
