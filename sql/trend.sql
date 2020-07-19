-- SQLite
select rdate, rprice, phigh, plow, ((plow+phigh)/2.0-rprice)/0.00137 as trend, prob
from pdata
where rdate>1594113009
ORDER BY rdate DESC
LIMIT 32;