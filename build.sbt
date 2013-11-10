import AssemblyKeys._ // put this at the top of the file

name := "breeze-examples"

version := "0.5.2"

organization := "org.scalanlp"

scalaVersion := "2.10.3"

resolvers ++= Seq(
  "ScalaNLP Maven2" at "http://repo.scalanlp.org/repo",
  "Scala Tools Snapshots" at "http://scala-tools.org/repo-snapshots/",
  "Typesafe Repository" at "http://repo.typesafe.com/typesafe/releases/",
   "Sonatype Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots/"
)

libraryDependencies ++= Seq(
  "junit" % "junit" % "4.5" % "test",
  "org.scalanlp" %% "breeze" % "0.5.2",
  "org.scalanlp" % "chalk" % "1.3.0" intransitive(),
  "org.scalanlp" % "nak" % "1.2.0" intransitive(),
  "org.scalanlp" %% "breeze-viz" % "0.5.2"
)



credentials += Credentials(Path.userHome / ".ivy2" / ".credentials")

javaOptions += "-Xmx3g"

seq(assemblySettings: _*)
