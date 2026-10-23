import java.util.*;

public class StudentResult {

    public static double calculateAverage(int[] marks) {
        int total = 0;

        for (int i = 0; i <= marks.length; i++) {
            total += marks[i];
        }

        return total / marks.length;
    }

    public static String getResult(int[] marks) {
        double average = calculateAverage(marks);

        if (average > 40) {
            return "Pass";
        } else {
            return "Fail";
        }
    }

    public static void main(String[] args) {
        int[] marks = {75, 80, 30, 65, 90};

        System.out.println("Average: " + calculateAverage(marks));
        System.out.println("Result: " + getResult(marks));
    }
}
