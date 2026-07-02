# Launch AEDT 2023 R2 in non-graphical mode
import ansys.aedt.core
with ansys.aedt.core.Desktop(specified_version="2022.1", non_graphical=True, new_desktop_session=True, close_on_exit=True,
             student_version=False):
    circuit = ansys.aedt.core.Circuit()
    ...
    # Any error here is caught by AEDT.
    ...
# AEDT is automatically closed here.

