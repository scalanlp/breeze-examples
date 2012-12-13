package breeze.classify
import breeze.linalg._
import breeze.plot._
import io.Source
import breeze.numerics._
import java.awt.{Color, Paint}
import breeze.stats.ContingencyStats


/**
 * example from a modified version by Shingo Omura (@everpeace). MIT Licensed.
 */
object GaussianKernelSVM {
  def main(arg: Array[String]) {
    // loading sample data
    val reg = "(-?[0-9]*\\.[0-9]+)\\,(-?[0-9]*\\.[0-9]+)\\,([01])*".r
    val data = DenseMatrix(Source.fromFile("data/SupportVectorMachineWithGaussianKernel.txt").getLines().toList.flatMap(_ match {
      case reg(x1, x2, y) => Seq((x1.toDouble, x2.toDouble, y.toDouble))
      case _ => Seq.empty
    }): _*)
    println("Data Loaded:\nX-value\tY-value\tResult(1=accepted/0=rejected)\n" + data)

    // plot sample
    val X = data(::, 0 to 1)
    val y = data(::, 2)
    val f = Figure()
    f.subplot(0) +=  scatter(X(::, 0), X(::, 1), {(_:Int) => 0.01}, y2Color(y))
    f.subplot(0).xlabel = "X-value"
    f.subplot(0).ylabel = "Y-value"
    f.subplot(0).title = "Input data"

    // learning parameter
    // C: regularized parameter
    // sigma: gaussian Kernel parameter
    val C = 1d
    val sigma = 0.1d

    // learn svm
    println("\n\npaused... press enter to start learning SVM.")
    readLine
    val model = trainSVM(X, y, C, gaussianKernel(sigma))
    val accr = ContingencyStats(y.toArray, predict(model)(X).toArray)
    println("\nTraining Statistics:" + accr)

    // plotting decision boundary
    println("paused... press enter to plot leaning result.")
    readLine
    plotDecisionBoundary(f.subplot(0), X, y, model)
    f.refresh()

    println("\n\nTo finish this program, close the result window.")
  }

  // gaussian kernel
  def gaussianKernel(sigma: Double)(x1: DenseVector[Double], x2: DenseVector[Double]): Double = {
    val diff = (x1 - x2)
    math.exp(-diff.dot(diff)/(2 * sigma * sigma))
  }

  // SVM Model
  case class Model(X: DenseMatrix[Double], y: DenseVector[Double], kernelF: (DenseVector[Double], DenseVector[Double]) => Double,
                   b: Double, alphas: DenseVector[Double], w: DenseVector[Double])

  // predict by SVM Model
  def predict(model: Model)(X: DenseMatrix[Double]): DenseVector[Double] = {
    val pred = DenseVector.zeros[Double](X.rows)
    val Xt = X.t
    val mxt = model.X.t
    for (i <- 0 until X.rows) {
      var prediction = 0d
      for (j <- 0 until model.X.rows) {
        prediction = prediction + model.alphas(j) * model.y(j) * model.kernelF(Xt(::, i), mxt(::, j))
      }
      pred(i) = I(prediction + model.b >= 0)
    }

    pred
  }

  // train SVM
  // This is a simplified version of the SMO algorithm for training SVMs.
  def trainSVM(X: DenseMatrix[Double], Y: DenseVector[Double], C: Double,
               kernel: (DenseVector[Double], DenseVector[Double]) => Double,
               tol: Double = 1e-3, max_passes: Int = 5): Model = {
    val m = X.rows
    val n = X.cols
    val Y2 = DenseVector.vertcat(Y)
    Y2 *= 2.0 -= 1.0 // remap 0 to -1
    val alphas = DenseVector.zeros[Double](m)
    var b = 0.0d
    val E = DenseVector.zeros[Double](m)
    var passes = 0
    var eta = 0.0d
    var L = 0.0d
    var H = 0.0d

    val Xt = X.t

    // generate Kernel DenseMatrix
    val K: DenseMatrix[Double] = DenseMatrix.zeros[Double](m, m)
    for (i <- 0 until m; j <- i until m) {
      K(i, j) = kernel(Xt(::, i), Xt(::, j))
      K(j, i) = K(i, j) // the matrix is symmetric.
    }

    print("Training(C=%f) (This takes a few minutes.)\n".format(C))
    var dots = 0
    while (passes < max_passes) {
      var num_alpha_changed = 0
      for (i <- 0 until m) {
        E(i) = b + (alphas :* (Y2 :* K(::, i))).sum - Y2(i)
        if ((Y2(i) * E(i) < -tol && alphas(i) < C) || (Y2(i) * E(i) > tol && alphas(i) > 0)) {
          var j = scala.math.ceil((m - 1) * scala.util.Random.nextDouble()).toInt
          // Make sure i \neq j
          while (j == i) (j = scala.math.ceil((m - 1) * scala.util.Random.nextDouble()).toInt)

          //Calculate Ej = f(x(j)) - y(j) using (2).
          E(j) = b + (alphas :* (Y2 :* K(::, j))).sum - Y2(j)

          //Save old alphas
          var alpha_i_old = alphas(i)
          var alpha_j_old = alphas(j)

          // Compute L and H by (10) or (11).
          if (Y2(i) == Y2(j)) {
            L = scala.math.max(0, alphas(j) + alphas(i) - C)
            H = scala.math.min(C, alphas(j) + alphas(i))
          } else {
            L = scala.math.max(0, alphas(j) - alphas(i))
            H = scala.math.min(C, C + alphas(j) - alphas(i))
          }

          //Compute eta by (14).
          eta = 2 * K(i, j) - K(i, i) - K(j, j)

          if (L != H && eta < 0) {
            //Compute and clip new value for alpha j using (12) and (15).
            alphas(j) = alphas(j) - (Y2(j) * (E(i) - E(j))) / eta

            //Clip
            alphas(j) = scala.math.min(H, alphas(j))
            alphas(j) = scala.math.max(L, alphas(j))

            // Check if change in alpha is significant
            if (math.abs(alphas(j) - alpha_j_old) < tol) {
              //continue to next i.
              // replace anyway
              alphas(j) = alpha_j_old
            } else {
              //Determine value for alpha i using (16).
              alphas(i) = alphas(i) + Y2(i) * Y2(j) * (alpha_j_old - alphas(j))

              //Compute b1 and b2 using (17) and (18) respectively.
              var b1 = b - E(i) - (Y2(i) * (alphas(i) - alpha_i_old) * K(i, j))
              -(Y2(j) * (alphas(j) - alpha_j_old) * K(i, j))
              var b2 = b - E(j) - (Y2(i) * (alphas(i) - alpha_i_old) * K(i, j))
              -(Y2(j) * (alphas(j) - alpha_j_old) * K(j, j))

              // Compute b by (19).
              if (0 < alphas(i) && alphas(i) < C) {
                b = b1
              } else if (0 < alphas(j) && alphas(j) < C) {
                b = b2
              } else {
                b = (b1 + b2) / 2.0d
              }

              num_alpha_changed += 1
            }
          }
        }
      }

      if (num_alpha_changed == 0) {
        passes += 1
      } else {
        passes = 0
      }

      print(".")
      dots += 1
      if (dots > 78) {
        print("\n")
        dots = 0
      }
    }
    print("Done! \n\n")

    val _idx:IndexedSeq[Int] = alphas.findAll(_ > 0.0d)
    val _X = X(_idx, ::).toDenseMatrix
    val _Y = Y2(_idx).toDenseVector
    val _kernel = kernel
    val _b = b
    val _alphas = alphas(_idx).toDenseVector
    val _w = ((alphas :* Y2).t * X) apply (::, 0)

    Model(_X, _Y, _kernel, _b, _alphas, _w)
  }

  def plotDecisionBoundary(plot: Plot, X: DenseMatrix[Double], y: DenseVector[Double], model: Model) = {
    print("Detecting decision boundaries...")
    // compute decision boundary.
    val NUM = 100
    val x1 = linspace(X(::, 0).min, X(::, 0).max, NUM)
    val x2 = linspace(X(::, 1).min, X(::, 1).max, NUM)
    val (bx1, bx2) = computeDecisionBoundary(x1, x2, predict(model))
    println(bx1, bx2)
    print(" Done!\n")

    // plot input data and detected boundary
    plot += scatter(X(::, 0), X(::, 1), {_ => 0.01}, y2Color(y))
    plot += scatter(bx1, bx2, {_ => 0.01}, { (_:Int) => Color.YELLOW})
    plot.xlabel = "X-value"
    plot.ylabel = "Y-value"
    plot.title = "Learning result by SVM\n blue:accepted, red: rejected, yellow:learned decision boundary"
    plot.refresh()
  }

  val i2color: Int => Paint = _ match {
    case 1 => Color.BLUE //accepted
    case 0 => Color.RED //rejected
    case _ => Color.BLACK //other
  }
  val y2Color: DenseVector[Double] => (Int => Paint) = y => {
    case i => i2color(y(i).toInt)
  }

  def meshgrid(x1: DenseVector[Double], x2: DenseVector[Double]): (DenseMatrix[Double], DenseMatrix[Double]) = {
    val x1Mesh = DenseMatrix.zeros[Double](x2.length, x1.length)
    for (i <- 0 until x2.length) {
      x1Mesh(i, ::) := x1.t
    }
    val x2Mesh = DenseMatrix.zeros[Double](x2.length, x1.length)
    for (i <- 0 until x1.length) {
      x2Mesh(::, i) := x2
    }
    (x1Mesh, x2Mesh)
  }


  def computeDecisionBoundary(x1: DenseVector[Double], x2: DenseVector[Double], predict: DenseMatrix[Double] => DenseVector[Double]): (DenseVector[Double], DenseVector[Double]) = {
    val (x1Mesh, x2Mesh) = meshgrid(x1, x2)
    val decisions = DenseMatrix.zeros[Double](x1Mesh.rows, x1Mesh.cols)

    // compute decisions for all mesh points.
    for (i <- 0 until x1Mesh.cols) {
      val this_X: DenseMatrix[Double] = DenseVector.horzcat(x1Mesh(::, i), x2Mesh(::, i))
      decisions(::, i) := predict(this_X)
    }

    // detect boundary.
    var bx1 = Seq[Double]()
    var bx2 = Seq[Double]()
    for (i <- 1 until decisions.rows - 1; j <- 1 until decisions.cols - 1) {
      if (decisions(i, j) == 0d && (decisions(i - 1, j - 1) == 1d || decisions(i - 1, j) == 1d || decisions(i - 1, j + 1) == 1d
        || decisions(i, j - 1) == 1d || decisions(i, j + 1) == 1d
        || decisions(i + 1, j) == 1d || decisions(i + 1, j) == 1d || decisions(i + 1, j + 1) == 1d)) {
        bx1 = x1Mesh(i, j) +: bx1
        bx2 = x2Mesh(i, j) +: bx2
      }
    }

    (DenseVector(bx1: _*), DenseVector(bx2: _*))
  }
}
