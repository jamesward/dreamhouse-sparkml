name := "dreamhouse-sparkml"

scalaVersion := "2.11.11"

libraryDependencies ++= Seq(
  "org.http4s"       %% "http4s-dsl"           % "0.17.5",
  "org.http4s"       %% "http4s-blaze-client"  % "0.17.5",
  "org.http4s"       %% "http4s-blaze-server"  % "0.17.5",
  "org.http4s"       %% "http4s-circe"         % "0.17.5",
  "io.circe"         %% "circe-generic"        % "0.8.0",
  "org.apache.spark" %% "spark-mllib"          % "2.2.0",
  "org.scalatest"    %% "scalatest"            % "3.0.4" % "test"
)

fork in run := true

enablePlugins(JavaAppPackaging)
