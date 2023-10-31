
The Ddl.G.Sch mode described above assumes that all tasks operate at their worst-case execution time. While this approach ensures the system's schedulability, it often results in a significant gap between a task's average-case performance and its worst-case performance. In the efficiency centralized real-time system, this gap can be leveraged to achieve even greater energy efficiency. By using techniques such as dynamic voltage and frequency scaling (DVFS) or task migration, a real-time system can take advantage of a task's lower than worst-case execution time to reduce energy consumption while still meeting its deadlines.

If we make the assumption that the tasks can miss deadlines once in a while manageably, the Ddl.G.Sch approach can be further optimized for more energy efficiency. There are two major pessemism we wish to resolve.

The $\Delta R_{CL_i^j}$ variable initially defined from the Figure \ref{fig:explanation_frequency_update} accomodated a pessemistic assumption, that the job $CL_i^j$ should start execution at the expected timestamp. In reality, there is a great possibility that the previous jobs $CL_i^0, PL_i^0, CL_i^1, PL_i^1, \cdots$ have finished execution early, giving job $CL_i^j$ chance to execute early. Therefore, the time saved for $CL_i^j$ to start execution early can be seen as part of $\Delta R_{CL_i^j}$

To utilize this optimization, as we see in Figure \ref{fig:explanation_frequency_update_effc}, $\Delta R_{CL_i^j}$ can be redefined as $\Delta R_{ECL_i^j}$ the absolute difference between the expected and real-time timestamp that job $CL_i^j$ finishes execution. Applying this optimization, we can possibly make $\Delta R_{ECL_i^j}$ larger, which further optimizes $f_i$. Implementation-wise, whenever a task \tau_i is released, all segments $CL_i^j$ will be assigned expected timestamps $R_{ECL_i^j}$ that they are supposed to finish according to the response time analysis in **II.A**

Another potential loss of efficiency due to pessemism is $f = max(f_0,f_1,...)$. While this ensures schedulability of all tasks, It also makes it so that whenever a task $\tau_i$ started its release, it will always assign the frequency $f_i=f_{max}$ until its first segment $CL_i^0$ has been finished. When a task set involves a lot of tasks, it is likely that the tasks with lower priorities will have their computation segment blocked for a long time, causing the frequency to keep at $f_{max}$ for a long time.

In Eff.C.Sch, we propose an enhancement to the Ddl.G.Sch approach. Namely, we make $f = avg(f_0,f_1,...)$. This way the earlier finished tasks can already have an impact on the system execution frequency. However in that case, the schedulability analysis for Ddl.G.Sch becomes invalid. Tasks may fail to schedule at runtime even if they have been passed by the response time analysis in **II.A**. In order to avoid consecutive deadline misses within the schedule,

If we keep applying the Eff.C.Sch approach throughout the task execution, there can be pitfalls where tasks keep dropping. In order to make the deadline misses manageable, we propose the mode switching policy between Eff.C.Sch and Ddl.G.Sch. In general, the execution starts at Eff.C.Sch. In case that a deadline miss happens, the task in execution will be dropped, and the system switches to Ddl.G.Sch until all tasks have successfully finished for once. To achieve this, each task, before they are released, will be assigned a "predefined expected response time". If by the time the task has not been finished, it will be aborted no matter it has started execution or not. Since all other tasks are still within the expected response time limit, from Lemma V.2, removing one task from the schedule will decrease the response time $\overline{R_{CL_i^j}}$ of all other tasks, so it will not invalidate the schedulability of the whole task as long as Ddl.G.Sch is applied.


% From Eq. (\ref{eq:frequency_update}), the $PL_i^j$ variable is assumed to be the worst-case execution time of job $PL_i^j$. In order to eliminate the pessemism, we can change $PL_i^j$ to $EPL_i^j$ to be the average-case execution time of job $PL_i^j$.