/*
Caroline Hsu - 4/10/2023

I created a neural Network to recognize 5x3 inputs. This implements the Forward Propagation and
Backward Propagation algorithms and implements a file reader. I have 15 input nodes,
12 hidden nodes, and 10 input nodes. 
 */
// package neuralnetwork;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Random;
import java.util.Arrays;

public class NeuralNetwork
{
  // declare hidden, input, and output size nodes
  public static int inputSize = 15;
  public static int hiddenSize = 12;
  public static int outputSize = 10;

  // declare the input and result data sets
  public static ArrayList<int[]> inputDataSet;
  public static ArrayList<int[]> resultDataSet;

  // method to create the hidden weights between -0.2 and 0.2
  public static ArrayList<double[]> createHiddenWeights()
  {
    // declares the hidden layer 
    ArrayList<double[]> hiddenLayer = new ArrayList<>();
    // generates a new random weight
    Random random = new Random();
    // loop through where the rows represent the input node number
    for (int r = 0; r < 15; r++)
    {
      double[] tempRow = new double[12];
      // loop through the columns as they represent the hidden node number
      for (int c = 0; c < 12; c++)
      {
        // random weight is added to the matrix
        tempRow[c] = random.nextDouble() * 0.4 - 0.2;
      }
      // add the temporary row to the hidden layer
      hiddenLayer.add(tempRow);
    }
    // and return the hidden layer
    return hiddenLayer;
  }

  // method that creates the output weights between -0.2 and 0.2
  public static ArrayList<double[]> createOutputWeights()
  {
    ArrayList<double[]> outputLayer = new ArrayList<>();
    // generates a new random weight
    Random random = new Random();
    // loop through where the rows represent the input node number
    for (int r = 0; r < 12; r++)
    {
      double[] tempRow = new double[10];
      // loop through where the columns represent the output node number
      for (int c = 0; c < 10; c++)
      {
        // random weight is added to the matrix
        tempRow[c] = random.nextDouble() * 0.4 - 0.2;
      }
      // add the temporary row to the hidden layer
      outputLayer.add(tempRow);
    }
    // and return the hidden layer
    return outputLayer;
  }

  // selects an input (uses string bc sample data is as string)
  public static String[] selectInput(ArrayList<String[]> sample)
  {
    // uses random to generate random integer in the sample size
    Random random = new Random();
    int randomVal = random.nextInt(sample.size());
    return sample.get(randomVal);
  }

  // selects an input (uses string bc validation data is a string)
  public static String[] selectValidationInput(ArrayList<String[]> validation)
  {
    // uses random to generate random integer in the validation size
    Random random = new Random();
    int randomVal = random.nextInt(validation.size());
    return validation.get(randomVal);
  }

  // calculates the sigmoid value of input list using formula given in the paper
  public static ArrayList<Double> calcInputSigmoid(String[] values)
  {
    ArrayList<Double> sigmoidList = new ArrayList<>();
    // loop through and usee the sigmoid formula
    for (int i = 0; i < values[0].length(); i++)
    {
      // use char at i bc of the string format
      sigmoidList.add((1.0) / (1.0 + (Math.exp(-Character.getNumericValue(values[0].charAt(i))))));
    }
    return sigmoidList;
  }

  // calculate the sigmoids value of list using the formula given in paper
  public static ArrayList<Double> calcSigmoid(ArrayList<Double> values)
  {
    ArrayList<Double> sigmoidList = new ArrayList<Double>();
    // loop through for the values size and use formula
    for (int i = 0; i < values.size(); i++)
    {
      double sigmoid = 1.0 / (1.0 + Math.exp(-values.get(i)));
      sigmoidList.add(sigmoid);
    }
    return sigmoidList;
  }

  // calculate the hidden nodes values given hdiden weights and input sigmoids
  public static ArrayList<Double> calcHiddenNodes(ArrayList<double[]> hiddenWeights, ArrayList<Double> inputSigmoid)
  {
    ArrayList<Double> hiddenNodes = new ArrayList<Double>();
    // loop through each value in hidden weights using rows and columns
    for (int c = 0; c < 12; c++)
    {
      double temp = 0;
      for (int r = 0; r < 15; r++)
      {
        // summation of input times weights for each hidden node 
        temp = temp + hiddenWeights.get(r)[c] * inputSigmoid.get(r);
      }
      hiddenNodes.add(temp);
    }
    return hiddenNodes;
  }

  // calculate the output nodes using the output weights and the hidden sigmoid
  public static ArrayList<Double> calcOutputNodes(ArrayList<double[]> outputWeights,
                                                  ArrayList<Double> hiddenSigmoid)
  {
    ArrayList<Double> outputNodes = new ArrayList<Double>();
    // loop through the output and hidden lengths
    for (int c = 0; c < 10; c++)
    {
      double temp = 0;
      for (int r = 0; r < 12; r++)
      {
        // summation of input times weights for each output node 
        temp = temp + outputWeights.get(r)[c] * hiddenSigmoid.get(r);
      }
      // add the temp to output nodes to return it
      outputNodes.add(temp);
    }
    return outputNodes;
  }

  // method calculates the output delta using output sigmoids and the expected output
  public static ArrayList<Double> calcOutputDelta(ArrayList<Double> outputSigmoid, String expectedOutput)
  {
    ArrayList<Double> outputDelta = new ArrayList<Double>();
    for (int i = 0; i < 10; i++)
    {
      double expOut = 0;
      // parse expected output into an integer, if expected output is same as node 
      // number, expected out is 1
      if (Integer.parseInt(expectedOutput) == i)
      {
        expOut = 1;
      }
      // use delta formula given in paper
      outputDelta.add(outputSigmoid.get(i) * (1 - outputSigmoid.get(i)) * (expOut - outputSigmoid.get(i)));
    }
    return outputDelta;
  }

  // calculate the output adjustment given the learning rates, output delta, and hidden sigmoids
  public static ArrayList<ArrayList<Double>> calcOutputAdjustment(double learningRate,
                                                                  ArrayList<Double> outputDelta,
                                                                  ArrayList<Double> hiddenSigmoid)
  {
    ArrayList<ArrayList<Double>> outputAdjustment = new ArrayList<ArrayList<Double>>();
    // loop through the hidden and output nodes length
    for (int r = 0; r < 12; r++)
    {
      ArrayList<Double> temp = new ArrayList<Double>();
      for (int c = 0; c < 10; c++)
      {
        // apply delta formula and add values to the output adjustment to return
        temp.add(learningRate * outputDelta.get(c) * hiddenSigmoid.get(r));
      }
      outputAdjustment.add(temp);
    }
    return outputAdjustment;
  }

  // adjust nodes using old weights and the adjustment 
  public static ArrayList<double[]> adjustNodes(ArrayList<double[]> oldWeights, ArrayList<ArrayList<Double>> adjustment)
  {
    ArrayList<double[]> newWeights = new ArrayList<>();
    // loop through old weights size and old weights size at the row length
    for (int r = 0; r < oldWeights.size(); r++)
    {
      double[] temp = new double[oldWeights.get(0).length];
      for (int c = 0; c < oldWeights.get(0).length; c++)
      {
        // add the adjustment to each weight
        temp[c] = oldWeights.get(r)[c] + adjustment.get(r).get(c);
      }
      newWeights.add(temp);
    }
    // return new weights for user to use
    return newWeights;
  }

  // calculate the hidden alpha given output delta and output weights
  public static ArrayList<Double> calcHiddenAlpha(ArrayList<Double> outputDelta, ArrayList<double[]> outputWeights)
  {
    ArrayList<Double> hiddenAlpha = new ArrayList<Double>();
    // double loop for hidden and output nodes length
    for (int r = 0; r < 12; r++)
    {
      double temp = 0;
      for (int c = 0; c < 10; c++)
      {
        // finds the summation of all the weights from hidden node to output node 
        // multiplied by the  delta output
        temp = temp + (outputWeights.get(r)[c] * outputDelta.get(c));
      }
      hiddenAlpha.add(temp);
    }
    // return the hidden alpha arraylist for users
    return hiddenAlpha;
  }

  // calculate the hidden delta given the hidden alpha and hidden sigmoid values
  public static ArrayList<Double> calcHiddenDelta(ArrayList<Double> hiddenAlpha, ArrayList<Double> hiddenSigmoid)
  {
    ArrayList<Double> hiddenDelta = new ArrayList<Double>();
    // loop through the hidden node length
    for (int i = 0; i < 12; i++)
    {
      // apply delta formula provided to get each value and return 
      hiddenDelta.add(hiddenSigmoid.get(i) * (1 - hiddenSigmoid.get(i)) * hiddenAlpha.get(i));
    }
    return hiddenDelta;
  }

  // calculate the hidden adjustment using the learning rate provided, hidden delta, and input sigmoids
  public static ArrayList<ArrayList<Double>> calcHiddenAdjustment(double learningRate,
                                                                  ArrayList<Double> hiddenDelta,
                                                                  ArrayList<Double> inputSigmoid)
  {
    ArrayList<ArrayList<Double>> hiddenAdjustment = new ArrayList<ArrayList<Double>>();
    // loop through input and hidden node lengths
    for (int r = 0; r < 15; r++)
    {
      ArrayList<Double> temp = new ArrayList<Double>();
      for (int c = 0; c < 12; c++)
      {
        // use the adjustment formula to find the values and add it to hidden adjustment 
        temp.add(learningRate * hiddenDelta.get(c) * inputSigmoid.get(r));
      }
      hiddenAdjustment.add(temp);
    }
    return hiddenAdjustment;
  }

  /*
  i didn't end up using this method as i found an eeasier method using buffered file
  reader, but i just kept this here in case i wanted to use it in the future. (sorry for 
  clunky code but i didn't feel like making a new project for this)
  // method that parses a line and stores the info into two arraylists for input and expected
  public static void parseLine(String line)
  {
    int[] inputData = new int[inputSize];
    int[] outputData = new int[outputSize];
    char lineChar;
    int inputIndex = 0;
    int outputIndex = 0;
    // split the line into two = input and output
    String[] splitLine = line.split(",");
    if (splitLine.length != 2)
    {
      System.out.println("Missing comma; line ignored");
    }
    else if (splitLine[0] == null || splitLine[1] == null)
    {
      System.out.println("Missing data; line ignored");
    }
    else
    {
      for (int i = 0; i < splitLine[0].length(); i++)
      {
        lineChar = splitLine[0].charAt(i);
        if (Character.isDigit(lineChar))
        {
          inputData[inputIndex] = Character.getNumericValue(lineChar);
          inputIndex++;
        }
      }
      // if else just ignore it
    }

    for (int i = 0; i < splitLine[1].length(); i++)
    {
      lineChar = splitLine[1].charAt(i);
      if (Character.isDigit(lineChar))
      {
        outputData[outputIndex] = Character.getNumericValue(lineChar);
        outputIndex++;
      }
    }
    if (inputIndex != inputSize)
    {
      System.out.println("Invalid input size, line is ignored");
    }
    else if (outputIndex != outputSize)
    {
      System.out.println("Invalid output size, line is ignored");
    }
    else
    {
      inputDataSet.add(inputData);
      resultDataSet.add(outputData);
    }
  }
   */
  // read file using the filepath that enters it into an arraylist of a string array
  public static ArrayList<String[]> readFile(String filePath) throws Exception
  {
    ArrayList<String[]> sampleData = new ArrayList<String[]>();
    try
    {
      BufferedReader br = new BufferedReader(new FileReader(filePath));
      String line = null;

      // split using the commas in the text files
      while ((line = br.readLine()) != null)
      {
        String[] values = line.split(",");
        // add the array of values into the arraylist to return to user
        sampleData.add(values);
      }
    }
    // throws a error for robustness (if there's an error getting the file from path name
    catch (Exception e)
    {
      System.out.println("Error getting file");
    }
    return sampleData;
  }

  // finds the expected output using the input validation data
  public static ArrayList<String[]> findExpectedOutput(ArrayList<String[]> input)
  {
    ArrayList<String[]> inputList = new ArrayList<String[]>();
    // iterates/loops through each value in the input arraylist of elements
    for (int i = 0; i < input.size(); i++)
    {
      String[] tempArray = new String[3];
      String expOut = "0";
      // loops over each character of the second element of the input array
      for (int j = 0; j < input.get(i)[1].length(); j++)
      {
        // if the current character is '1'
        if (input.get(i)[1].charAt(j) == '1')
        {
          // set the expected output to the index of this character
          expOut = Integer.toString(j);
          // and exit da loop
          break;
        }
      }
      // set input array equal to temp array to add to input list (corresponding elements)
      for (int j = 0; j < 4; j++)
      {
        tempArray[0] = input.get(i)[0];
        tempArray[1] = input.get(i)[1];
        tempArray[2] = expOut;
      }
      inputList.add(tempArray);
    }
    // return input list to user of expected outputs
    return inputList;
  }

  public static void main(String[] args)
  { /*
    this was before i started using the file reader, i just kept it here so i didn't
    have to delete it in case the file reader didn't work (but it ended up working)
    String[][] trainingData = {
        {"111101101101111", "1000000000", "0"},
        {"001001001001001", "0100000000", "1"},
        {"001001001001000", "0100000000", "1"},
        {"000001001001001", "0100000000", "1"},
        {"111001111100111", "0010000000", "2"},
        {"011001111100111", "0010000000", "2"},
        {"111001111100110", "0010000000", "2"},
        {"111001111001111", "0001000000", "3"},
        {"111001011001111", "0001000000", "3"},
        {"011001111001111", "0001000000", "3"},
        {"111001111001011", "0001000000", "3"},
        {"101101111001001", "0000100000", "4"},
        {"001101111001001", "0000100000", "4"},
        {"100101111001001", "0000100000", "4"},
        {"101101111001000", "0000100000", "4"},
        {"111100111001111", "0000010000", "5"},
        {"110100111001111", "0000010000", "5"},
        {"111100111001011", "0000010000", "5"},
        {"111100111101111", "0000001000", "6"},
        {"110100111101111", "0000001000", "6"},
        {"100100111101111", "0000001000", "6"},
        {"111001001001001", "0000000100", "7"},
        {"111001001001000", "0000000100", "7"},
        {"111101111101111", "0000000010", "8"},
        {"111101111001111", "0000000001", "9"},
        {"111101111001011", "0000000001", "9"},
        {"111101111001001", "0000000001", "9"}
    };
    String[][] validationData = {
    {"111101101101111","1000000000","0"},
    {"000001001001001","0100000000","1"},
    {"000001001001000","0100000000","1"},
    {"000001001001001","0100000000","1"},
    {"111001111100111","0010000000","2"},
    {"111001111100110","0010000000","2"},
    {"011001111100110","0010000000","2"},
    {"111001111001111","0001000000","3"},
    {"111001011001111","0001000000","3"},
    {"011001011001011","0001000000","3"},
    {"101101111001001","0000100000","4"},
    {"001101111001001","0000100000","4"},
    {"001101111001000","0000100000","4"},
    {"111100111001111","0000010000","5"},
    {"110100111001111","0000010000","5"},
    {"110100111001011","0000010000","5"},
    {"111100111101111","0000001000","6"},
    {"110100111101111","0000001000","6"},
    {"100100111101111","0000001000","6"},
    {"111001001001001","0000000100","7"},
    {"111001001001000","0000000100","7"},
    {"011001001001001","0000000100","7"},
    {"111101111101111","0000000010","8"},
    {"111101111001111","0000000001","9"},
    {"111101111001011","0000000001","9"},
    {"111101111001001","0000000001","9"}
    };  
    */
    
    // to read sample data and validation data files 
    ArrayList<String[]> trainingData = new ArrayList<String[]>();
    ArrayList<String[]> validationData = new ArrayList<String[]>();
    try
    {
      trainingData = readFile("/Users/carolinehsu/NetBeansProjects/NeuralNetwork/src/neuralnetwork/SampleData.txt");
      validationData = readFile("/Users/carolinehsu/NetBeansProjects/NeuralNetwork/src/neuralnetwork/ValidationData.txt");
    }
    catch (Exception e)
    {
      System.out.println("error with file");
    }

    // use expected output methods to find the expected outputs
    trainingData = findExpectedOutput(trainingData);
    validationData = findExpectedOutput(validationData);

    // declare iteration size and learning rates
    double iterationSize = 28000;
    double learningRate = .33;

    // createe weights for output and hidden
    ArrayList<double[]> hiddenWeights = createHiddenWeights();
    ArrayList<double[]> outputWeights = createOutputWeights();

    // loop through thee itreation size
    for (int i = 0; i < iterationSize; i++)
    {
      // forward propagation
      String[] inputNodes = selectInput(trainingData);
      ArrayList<Double> inputSigmoid = calcInputSigmoid(inputNodes);
      ArrayList<Double> hiddenNodes = calcHiddenNodes(hiddenWeights, inputSigmoid);
      ArrayList<Double> hiddenSigmoid = calcSigmoid(hiddenNodes);
      ArrayList<Double> outputNodes = calcOutputNodes(outputWeights, hiddenSigmoid);
      ArrayList<Double> outputSigmoid = calcSigmoid(outputNodes);

      // back propagation
      ArrayList<Double> outputDelta = calcOutputDelta(outputSigmoid, inputNodes[2]);
      ArrayList<ArrayList<Double>> outputAdjustment = calcOutputAdjustment(learningRate,
          outputDelta, hiddenSigmoid);
      outputWeights = adjustNodes(outputWeights, outputAdjustment);
      ArrayList<Double> hiddenAlpha = calcHiddenAlpha(outputDelta, outputWeights);
      ArrayList<Double> hiddenDelta = calcHiddenDelta(hiddenAlpha, hiddenSigmoid);
      ArrayList<ArrayList<Double>> hiddenAdjustment = calcHiddenAdjustment(learningRate, hiddenDelta, inputSigmoid);
      hiddenWeights = adjustNodes(hiddenWeights, hiddenAdjustment);
    }

    // caclulate the accuracy of all the validation data using forward propagation
    double accuracyCount = 0;
    for (int i = 0; i < validationData.size(); i++)
    {
      String[] stringInputNodes = validationData.get(i);
      String[] inputNodes = new String[stringInputNodes.length];
      // copy array
      for (int k = 0; k < stringInputNodes.length; k++)
      {
        inputNodes[k] = stringInputNodes[k];
      }
      // forward propagation for validation data
      ArrayList<Double> inputSigmoid = calcInputSigmoid(inputNodes);
      ArrayList<Double> hiddenNodes = calcHiddenNodes(hiddenWeights, inputSigmoid);
      ArrayList<Double> hiddenSigmoid = calcSigmoid(hiddenNodes);
      ArrayList<Double> outputNodes = calcOutputNodes(outputWeights, hiddenSigmoid);
      ArrayList<Double> outputSigmoid = calcSigmoid(outputNodes);
      int best = 0;
      // finds highest output sigmoid value node number
      double maxOutput = outputSigmoid.get(0);
      for (int j = 1; j < outputSigmoid.size(); j++)
      {
        if (outputSigmoid.get(j) > maxOutput)
        {
          maxOutput = outputSigmoid.get(j);
          best = j;
        }
      }
      // prints the input, expected output, and the result of the number it actually found
      System.out.println("Input: " + inputNodes[0]);
      System.out.println("Expected Output: " + inputNodes[1]);
      System.out.println("Result: " + best);
      System.out.println();

      // if output is the expected output then increment the accuracy count
      if (best == Integer.parseInt(inputNodes[2]))
      {
        accuracyCount++;
      }
    }
    // display the accuracy count as a percent
    System.out.println("Accuracy: " + ((accuracyCount / 26) * 100));
    // i am done with the program yay :) if you read this i hope you have a nice day
  }
}
