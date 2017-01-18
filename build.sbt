name := "dreamhouse-sparkml"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.http4s"       %% "http4s-dsl"           % "0.15.3",
  "org.http4s"       %% "http4s-blaze-client"  % "0.15.3",
  "org.http4s"       %% "http4s-blaze-server"  % "0.15.3",
  "org.http4s"       %% "http4s-circe"         % "0.15.3",
  "io.circe"         %% "circe-generic"        % "0.6.1",
  "org.apache.spark" %% "spark-mllib"          % "2.1.0",
  "org.scalatest"    %% "scalatest"            % "3.0.1" % "test"
)

fork in run := true

enablePlugins(JavaAppPackaging)
