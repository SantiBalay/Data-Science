DROP PROCEDURE IF EXISTS gender_department_breakdown_by_range;

DELIMITER $$
create procedure gender_department_breakdown_by_range (IN lower float, IN higher float) 
begin
	SELECT 
    e.gender as gender, AVG(s.salary) as average_salary , d.dept_name as department_name
FROM
    t_salaries s
        JOIN
    t_employees e ON s.emp_no = e.emp_no
        JOIN
    t_dept_emp td ON td.emp_no = e.emp_no
        JOIN
    t_departments d ON td.dept_no = d.dept_no
WHERE
    s.salary BETWEEN lower AND higher
GROUP BY e.gender , d.dept_name;
end$$

DELIMITER ;

CALL gender_department_breakdown_by_range(50000,90000);

