SELECT YEAR(s.from_date) as calendar_year,e.gender,ROUND(AVG(salary),2), d.dept_name
from t_salaries s JOIN t_employees e  ON(e.emp_no = s.emp_no) JOIN t_dept_emp t ON (t.emp_no = e.emp_no) JOIN t_departments d ON (d.dept_no = t.dept_no)
group by calendar_year, e.gender, d.dept_no
having calendar_year <= 2002
ORDER BY calendar_year, e.gender,d.dept_no