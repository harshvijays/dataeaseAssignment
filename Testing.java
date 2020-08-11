package com.mavenproject;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;
import java.sql.Connection;
import java.sql.DriverManager;
import java.sql.PreparedStatement;
import java.sql.ResultSet;
import java.sql.SQLException;

import java.sql.Statement;

import org.apache.commons.math3.stat.inference.TTest;

import com.opencsv.CSVWriter;

public class Testing {
	
	
	//Function to write output to csv file
	public static void writingToCsv(String filepath,String[] header,String[] data) throws IOException
	{
		 File file = new File(filepath); 
         FileWriter outputfile = new FileWriter(file); 
         CSVWriter writer = new CSVWriter(outputfile); 
         writer.writeNext(header);
         writer.writeNext(data);
         writer.close(); 

	}
	
	
	//Function to insert data in table from csv file 
	public static void insertingData(String file,Connection con) throws SQLException, IOException
    {
		
	String sql = "INSERT INTO CONFIDENTIAL_TABLE (ID , Isconfidential , ProjectName , Street , City , State , Zipcode ,Country ,LEEDSystemVersionDisplayName , PointsAchieved , CertLevel , CertDate ,IsCertified , OwnerTypes , GrossSqFoot , TotalPropArea , ProjectTypes ,OwnerOrganization , RegistrationDate ) VALUES (?, ?,?, ?, ?, ?,?,?,?,?,?,?,?,?,?,?,?,?,?)";
    PreparedStatement statement = con.prepareStatement(sql);
    BufferedReader lineReader = new BufferedReader(new FileReader(file));
    String lineText = null;

    
	lineReader.readLine();	// skip header line

    while ((lineText = lineReader.readLine()) != null) {
        String[] data = lineText.split(",");
        String ID = data[0];
        String Isconfidential = data[1];
        String ProjectName = data[2];
        String Street = data[3];
        String City = data[4];	                
        String State = data[5];
        String Zipcode = data[6];
        String Country = data[7];
        String LEEDSystemVersionDisplayName = data[8];
        String PointsAchieved  = data[9];
        String CertLevel = data[10];
        String CertDate = data[11];
        String IsCertified = data[12];
        String OwnerTypes = data[13];
        String GrossSqFoot = data[14];
        String TotalPropArea = data[15];
        String ProjectTypes = data[16];
        String OwnerOrganization = data[17];
        String RegistrationDate = data[18];

        statement.setString(1, ID);
        statement.setString(2, Isconfidential);
        statement.setString(3, ProjectName);
        statement.setString(4, Street);
        statement.setString(5, City);
        statement.setString(6, State);
        statement.setString(7, Zipcode);
        statement.setString(8, Country);
        statement.setString(9, LEEDSystemVersionDisplayName);
        statement.setString(10, PointsAchieved);
        statement.setString(11, CertLevel);
        statement.setString(12, CertDate);
        statement.setString(13, IsCertified);
        statement.setString(14, OwnerTypes);
        statement.setString(15, GrossSqFoot);
        statement.setString(16, TotalPropArea);
        statement.setString(17, ProjectTypes);
        statement.setString(18, OwnerOrganization);
        statement.setString(19, RegistrationDate);

        statement.executeUpdate();

    }

    lineReader.close();
    con.commit();
}
	
	// Main 
	public static void main(String[] args) {
		   
	      Connection con = null;
	      Statement stmt = null;
	      

	      try {
	         //Registering the HSQLDB JDBC driver
	         Class.forName("org.hsqldb.jdbc.JDBCDriver");
	         //Creating the connection with HSQLDB
	         con = DriverManager.getConnection("jdbc:hsqldb:mem:Test", "SA", "");
	         System.out.println("checking");
	         if (con!= null){
	            System.out.println("Connection created successfully");
	            
	         }else{
	            System.out.println("Problem with creating connection");
	         }
	         
	         stmt =  con.createStatement();
	         
	         String createst="CREATE TABLE confidential_table( ID VARCHAR(100), \n" + 
	         		"Isconfidential VARCHAR(100),\n" + 
	         		" ProjectName VARCHAR(100), \n" + 
	         		"Street VARCHAR(100),\n" + 
	         		" City VARCHAR(100),\n" + 
	         		" State VARCHAR(100),\n" + 
	         		" Zipcode VARCHAR(100),\n" + 
	         		" Country VARCHAR(100),\n" + 
	         		" LEEDSystemVersionDisplayName VARCHAR(100),\n" + 
	         		" PointsAchieved VARCHAR(100),\n" + 
	         		" CertLevel VARCHAR(100),\n" + 
	         		" CertDate VARCHAR(100),\n" + 
	         		" IsCertified VARCHAR(100), \n" + 
	         		"OwnerTypes VARCHAR(100),\n" + 
	         		" GrossSqFoot VARCHAR(100), \n" + 
	         		"TotalPropArea VARCHAR(100),\n" + 
	         		" ProjectTypes VARCHAR(100), \n" + 
	         		"OwnerOrganization VARCHAR(100),\n" + 
	         		" RegistrationDate VARCHAR(100),\n" + 
	         		"PRIMARY KEY (id) );";
	         stmt.executeUpdate(createst);
	      
	         
	         	//inserting data in hsql table 
	            insertingData(args[1], con);
	            insertingData(args[2], con);
	            
	           
	            
			/*
			 * ResultSet rs=stmt.executeQuery("select count(*) from CONFIDENTIAL_TABLE");
			 * rs.next(); int rowCount = rs.getInt(1); System.out.println(rowCount);
			 */     
	         
	            String query1="SELECT count(*) from "
	            		+ "(select LEEDSystemVersionDisplayName FROM confidential_table "
	            		+ " where State='Virginia' or State = 'VA'"
	            		+ "group by LEEDSystemVersionDisplayName)";
	            		
	            
	            String query2="SELECT count(*) from"
	            		+ "(select OwnerTypes from confidential_table"
	            		+ " where State='Virginia' or State = 'VA'"
	            		+ " group by OwnerTypes)";
	            		
	            String query3="select GrossSqFoot from confidential_table"
	            		+ " where (State='Virginia' or State = 'VA') and CertLevel='Certified'";
	            
	            String query4="select Zipcode , count(Zipcode) from  confidential_table "
	            		+ "where State='Virginia' or State='VA' "
	            		+ "group by Zipcode "
	            		+ "order by Count(Zipcode) desc "
	            		+ "limit 2 "; 
	            
	            String query5_1= "select PointsAchieved  from confidential_table "
	            		+ "where (State='Virginia' or State = 'VA') and LEEDSystemVersionDisplayName='LEED-NC 2.2'";
	            
	            String query5_2=" select PointsAchieved  from confidential_table"
	            		+ " where (State='California' or State='CA') and LEEDSystemVersionDisplayName='LEED-NC 2.2'";
	            
	            ResultSet rs1=stmt.executeQuery(query1);
	            rs1.next();
	            System.out.println("answer 1 = "+rs1.getInt(1));
	            String[] s1= {"No. of LEED Projects in Virginia"};
	            String[] d1= {Integer.toString(rs1.getInt(1))};
	            writingToCsv(args[0]+"1.csv",s1,d1);
	           
	            ResultSet rs2=stmt.executeQuery(query2);
	            rs2.next();
	            System.out.println("answer 2 = "+rs2.getInt(1));
	            String[] s2= {"No. of LEED Projects ownerTypes in Virginia"};
	            String[] d2= {Integer.toString(rs2.getInt(1))};
	           
	            writingToCsv(args[0]+"2.csv", s2,d2);
 
	            
	            float sum= 0;
	            ResultSet rs3=stmt.executeQuery(query3);
	            rs3.next();
	            //  int i=0,j=0;
	            while(rs3.next())
	            { 
		           String s=rs3.getString(1);
	               System.out.println("row number = "+rs3.getRow() );
	               try {
	            	   		sum+=Float.parseFloat((s.trim()));
	            		}
	               catch(NumberFormatException e) {
	            			rs3.next();
	            		}
	            }
	            System.out.println("answer 3 = "+sum);
	            String[] s3= {"Total GrossSqFoot"};
	            String[] d3= {Float.toString(sum)};
	           
	            writingToCsv(args[0]+"3.csv", s3,d3);

	           
	            ResultSet rs4=stmt.executeQuery(query4);
	            rs4.next();
	            System.out.println("answer 4 = "+rs4.getString(1)+"---value is "+rs4.getInt(2));
	            String[] s4= {"Zipcode","NO of Projects"};
	            String[] d4= {rs4.getString(1),Integer.toString(rs4.getInt(2))};
	           
	            writingToCsv(args[0]+"4.csv", s4,d4);
	           
	           
	            ResultSet rs5_1=stmt.executeQuery(query5_1);
	            rs5_1.next();
	            int i = 0;  
	            double[] ar1=new double[1200];
	            while (rs5_1.next()) {  
	            	try {
	            			ar1[i] = Double.parseDouble(rs5_1.getString(1));  
	            		}
	            	catch(NumberFormatException e)
	            		{
	            			rs5_1.next();
	            		}
	            	i++;  
	            	} 
	            System.out.println("value of i "+i);
	            
	            i=0;
	            ResultSet rs5_2=stmt.executeQuery(query5_2);
			    double ar2[]=new double[1200];  
	            while (rs5_2.next()) {
	            	try {
	            			ar2[i] = Double.parseDouble(rs5_2.getString(1));  
	            		}
	            	catch(NumberFormatException e )
	            		{
	            			rs5_2.next();
	            		}
	            	i++;  
	            	}  

	            System.out.println("value of i "+i);
	            TTest test = new TTest();
	            String nullhyp,infer;
	            double p= test.tTest(ar1,ar2);
	            System.out.println("value of p is  : "+p);
	            if (p>=0.05)
	            {     
	            	nullhyp="Rejected";
	            	infer="null hypothesis is rejected and there is a significant difference between 2 datasets ";
	            }
	            else
	            {
	            	nullhyp="Accepted";
	            	infer="null hypothesis failed to reject and there is NO significant difference between 2 datasets ";
	            }
	            String[] s5= {"p value","Null Hypothesis","Inference"};
		        String[] d5= {Double.toString(p),nullhyp,infer};
		           
	            writingToCsv(args[0]+"5.csv", s5,d5);
 
	            //closing the connection            
	            con.close();
	            
	            
	            
	      }  catch (Exception e) {
	         e.printStackTrace(System.out);
	      }
	   }
	}

