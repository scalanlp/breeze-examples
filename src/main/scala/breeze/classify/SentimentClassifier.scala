package breeze.classify

import java.io.File
import breeze.config.{GenerateHelp, Help, CommandLineParser}
import io.Source
import breeze.data.Example
import breeze.util.Index
import breeze.linalg._
import breeze.stats.ContingencyStats
import breeze.optimize.FirstOrderMinimizer.OptParams
import collection.mutable.ArrayBuffer

/**
 * Example showing how to build a simple sentiment classifier
 * in under 100 lines of code.
 *
 * It's set up to look at the following data:
 * http://www.cs.cornell.edu/people/pabo/movie-review-data/review_polarity.tar.gz
 * from this website: http://www.cs.cornell.edu/people/pabo/movie-review-data/
 *
 *
 * @author dlwh
 */
object SentimentClassifier {

  case class Params(@Help(text="Path to txt_sentoken in the dataset.") train:File = null,
                    help: Boolean = false)

  def main(args: Array[String]) {
    // Read in parameters, ensure they're right and dump to file if necessary
    val (config,seqArgs) = CommandLineParser.parseArguments(args)
    val params = try {
      config.readIn[Params]("")
    } catch {
      case e =>
        e.printStackTrace()
        Params(null,help=true)
    }
    if(params.help || params.train == null) {
      println(GenerateHelp[Params](config))
      sys.exit(1)
    }

    // get a tokenizer and a stemmer.
    val langData = breeze.text.LanguagePack.English
    val tokenizer = langData.simpleTokenizer
    val stemmer = langData.stemmer.getOrElse(identity[String] _)


    println("Reading in data...")
    val data: Array[Example[Int, IndexedSeq[String]]] = {
      for{ dir <- params.train.listFiles(); f <- dir.listFiles()} yield {
        // data is already sentence tokenized, just have to make word tokens
        val text = tokenizer(Source.fromFile(f).mkString).toIndexedSeq
        // data is in pos/ and neg/ directories
        Example(label=if(dir.getName =="pos") 1 else 0, features = text, id = f.getName)
      }
    }


    // These are the feature templates we use. We can use any we want.
    sealed trait Feature
    case class WordFeature(w: String) extends Feature
    case class StemFeature(w: String) extends Feature

    // We're going to use SparseVector representations of documents.
    // An Index maps  Features to Ints and back again.
    val featureIndex = Index[Feature]()

    // make two passes over the data, one to figure out how many features
    // we have, and then one to build the vectors.
    def extractFeatures(ex: Example[Int, IndexedSeq[String]]) =  {
      ex.map { words =>
        val builder = new VectorBuilder[Double](Int.MaxValue)
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
        builder.length = featureIndex.size
        builder.toSparseVector
      }
    }


    // do cross validation. Files are named cvXy_... where cvX is the fold for testing.
    // group by first 3 letters of id: cv0..., cv1,...
    val folds = extractedData.groupBy(_.id.take(3)).mapValues(_.toSet)
    val allStats = ArrayBuffer[ContingencyStats[Int]]()
    for( (nameOfFold, test) <- folds) {
      val train = extractedData.filterNot(test)
      // Optimization params:
      // do 50 passes with OWLQN, using L1 regularization
      // L1 regularization ensures that the resulting model is sparse,
      // which gives more interpretable models and usually helps performance.
      // Setting it to false will use L2 regularization, which doesn't
      // work as well on this dataset.
      // If you useStochastic, be sure to up the number of iterations!
      // (OWLQN is just LBFGS with special tricks to handle L1 regularization.)
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
