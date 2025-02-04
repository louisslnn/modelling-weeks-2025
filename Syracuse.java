import java.util.*;

public class Syracuse{

    public static List<Integer> getSyracuse(int integer){
        List<Integer> arr = new ArrayList<>();

        while (integer > 1){
            if (integer % 2 == 0){
                integer /= 2;
            }
            else{
                integer *= 3 + 1;
            }
            arr.add(integer);
        }
        return arr;
    }

    public static int sumSyracuse(int integer){
        List<Integer> sequence = getSyracuse(integer);
        int sum = 0;

        for (int n: sequence){
            sum += n;
        }
        return sum;
    }

    public static String getLargestPath(int N) {
        int largestPath = 0;
        int index = 0;

        for (int i = 1; i <= N; i++) {
            int pathLength = getSyracuse(i).size();
            if (pathLength > largestPath) {
                largestPath = pathLength;
                index = i; // Directly store i instead of using indexOf later
            }
        }

        return "The largest Syracuse sequence within 1 and " + N + " has a length of " + largestPath + 
               " steps, which corresponds to " + index;
    }

    public static void main(String[] args) {
        System.out.println("Program started...");
        System.out.println("Syracuse sequence of 871: " + getSyracuse(871));
        System.out.println("Sum of sequence for 871: " + sumSyracuse(871));
        System.out.println(getLargestPath(1000));
        System.out.println("Program finished.");
    }
}