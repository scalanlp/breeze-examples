package breeze.inference

/**
 * See http://darrenjw.wordpress.com/2013/10/04/a-functional-gibbs-sampler-in-scala/
 *
 * @author Darren J Wilkinson
 **/
object NormalGammaGibbsSampler {

  import breeze.stats.distributions._
  import scala.math.sqrt

  class State(val x: Double,val y: Double)

  def nextIter(s: State): State = {
    val newX=Gamma(3.0,1.0/((s.y)*(s.y)+4.0)).draw()
    new State(newX,Gaussian(1.0/(newX+1),1.0/sqrt(2*newX+2)).draw())
  }

  def nextThinnedIter(s: State,left: Int): State = {
    if (left==0) s
    else nextThinnedIter(nextIter(s),left-1)
  }

  def genIters(s: State,current: Int,stop: Int,thin: Int): State = {
    if (!(current>stop)) {
      println(current+" "+s.x+" "+s.y)
      genIters(nextThinnedIter(s,thin),current+1,stop,thin)
    }
    else s
  }

  def main(args: Array[String]) {
    println("Iter x y")
    genIters(new State(0.0,0.0),1,50000,1000)
  }


}
