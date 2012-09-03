package breeze.classify

import java.io.File
import breeze.config.CommandLineParser
import io.Source
import breeze.data.Example
import breeze.util.Index
import breeze.linalg._
import breeze.stats.ContingencyStats
import breeze.optimize.FirstOrderMinimizer.OptParams
import collection.mutable.ArrayBuffer

/**
 *
 * @author dlwh
 */
object SentimentClassifier {


  case class Params(train:File)

  def main(args: Array[String]) {
    val (config,seqArgs) = CommandLineParser.parseArguments(args)
    val params = config.readIn[Params]("")

    val langData = breeze.text.LanguagePack.English
    val tokenizer = langData.simpleTokenizer
    val stemmer = langData.stemmer.getOrElse(identity[String] _)


    val data: Array[Example[Int, IndexedSeq[String]]] = for{ dir <- params.train.listFiles(); f <- dir.listFiles()} yield {
      // data is already sentence tokenized, just have to make word tokens
      val text = tokenizer(Source.fromFile(f).mkString).toIndexedSeq
      // data is in pos/ and neg/ directories
      Example(label=if(dir.getName =="pos") 1 else 0, features = text, id = f.getName)
    }


    sealed trait Feature
    case class WordFeature(w: String) extends Feature
    case class StemFeature(w: String) extends Feature

    val featureIndex = Index[Feature]()

    // make two passes over the data, one to figure out how many features
    // we have, and then one to build the vectors.
    def extractFeatures(ex: Example[Int, IndexedSeq[String]]) =  {
      ex.map { words =>
        val builder = new SparseVector.Builder[Double](Int.MaxValue)
        for(w <- words) {
          val fi = featureIndex.index(WordFeature(w))
          val si = featureIndex.index(StemFeature(stemmer(w)))
          builder.add(fi, 1.0)
          builder.add(si, 1.0)
        }

        builder
      }
    }

    // second pass to build the sparse vectors, basically just set the right size and
    // get the resulting sparsevector
    val extractedData: IndexedSeq[Example[Int, SparseVector[Double]]] = data.map(extractFeatures).map{ex =>
      ex.map{ builder =>
        builder.dim = featureIndex.size
        builder.result()
      }
    }


    // do cross validation. Files are named cvXy_... where cvX is the fold for testing.
    // group by first 3 letters of id: cv0..., cv1,...
    val folds = extractedData.groupBy(_.id.take(3)).mapValues(_.toSet)
    val allStats = ArrayBuffer[ContingencyStats[Int]]()
    for( (nameOfFold, test) <- folds) {
      val train = extractedData.filterNot(test)
      val opt = OptParams(maxIterations=60,useStochastic=false,useL1=true)
      val classifier = new LogisticClassifier.Trainer[Int, SparseVector[Double]](opt).train(train)

      val stats = ContingencyStats(classifier, test.toSeq)
      allStats += stats
      println(stats)
    }

    for( (stats,i) <- allStats.zipWithIndex) {
      println("CV Fold " + i)
      println(stats)
    }


  }


}
