name := "dreamhouse-sparkml"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.http4s"       %% "http4s-dsl"          % "0.14.1a",
  "org.http4s"       %% "http4s-blaze-server" % "0.14.1a",
  "org.apache.spark" %% "spark-mllib"         % "1.6.2",
  "org.scalatest"    %% "scalatest"           % "2.2.6" % "test"
)

cancelable in Global := true

enablePlugins(JavaAppPackaging)
