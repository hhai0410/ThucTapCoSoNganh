"""End-to-end API checks against an isolated SQLite database, using stdlib only."""
import json, os, subprocess, sys, tempfile, time, unittest, urllib.request, urllib.error, socket, zipfile, io
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]

class SystemTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.url_db = "sqlite:///" + str(Path(cls.temp.name) / "test.db")
        with socket.socket() as sock:
            sock.bind(("127.0.0.1", 0)); cls.port = sock.getsockname()[1]
        env = dict(os.environ, DATABASE_URL=cls.url_db, JWT_SECRET_KEY="isolated-test-secret", PYTHONUTF8="1", DEMO_MODE="1")

        env["PYTHONPATH"] = os.pathsep.join(str(p) for p in sys.path if p)
        cls.proc = subprocess.Popen([getattr(sys, "_base_executable", sys.executable), "-m", "uvicorn", "backend.app.main:app", "--port", str(cls.port)], cwd=ROOT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        for _ in range(120):
            try:
                if cls.call("/health")[0] == 200: return
            except OSError: pass
            if cls.proc.poll() is not None: break
            time.sleep(.5)
        cls.proc.terminate(); cls.proc.wait()
        raise RuntimeError("Test server did not become ready")

    @classmethod
    def tearDownClass(cls):
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(cls.proc.pid), "/T", "/F"], capture_output=True)
        else:
            cls.proc.terminate()
        cls.proc.wait(timeout=15)
        for attempt in range(20):
            try:
                cls.temp.cleanup()
                break
            except PermissionError:
                if attempt==19: raise
                time.sleep(.25)

    @classmethod
    def call(cls, path, method="GET", body=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token: headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(f"http://127.0.0.1:{cls.port}" + path, data=json.dumps(body).encode() if body is not None else None, headers=headers, method=method)
        try:
            with urllib.request.urlopen(req) as r:
                raw=r.read()
                return r.status, json.loads(raw) if "json" in r.headers.get("Content-Type", "") else raw
        except urllib.error.HTTPError as e: return e.code, json.loads(e.read())

    def test_complete_flow(self):
        register={"full_name":"Test User", "email":"test@example.com", "password":"test1234"}
        self.assertEqual(self.call("/auth/register", "POST", dict(register,email="user@com"))[0],422)
        self.assertEqual(self.call("/auth/register", "POST", dict(register,role="admin"))[0],422)
        self.assertEqual(self.call("/auth/register", "POST", register)[0],201)
        self.assertEqual(self.call("/auth/register", "POST", register)[0],400)
        token=self.call("/auth/login","POST",register)[1]["access_token"]
        self.assertEqual(self.call("/history")[0],401)
        self.assertEqual(self.call("/history",token="bad")[0],401)
        self.assertEqual(self.call("/doctor/patients",token=token)[0],403)
        self.assertEqual(self.call("/history",token=token)[1],[])
        sample=dict(age=65,gender="Male",hypertension=1,heart_disease=0,ever_married="Yes",work_type="Private",Residence_type="Urban",avg_glucose_level=200,bmi=28.1,smoking_status="formerly smoked")
        self.assertEqual(self.call("/predict","POST",dict(sample,bmi=9),token)[0],422)
        self.assertEqual(self.call("/predict","POST",dict(sample,age=121),token)[0],422)
        start=time.perf_counter(); status,pred=self.call("/predict","POST",sample,token); elapsed=time.perf_counter()-start
        self.assertEqual(status,200,pred); self.assertTrue(pred["shap"])
        self.assertEqual(pred["prediction_result"],int(pred["risk_probability"]>=pred["threshold"]))
        print("Prediction latency:",round(elapsed,3),"seconds")
        h=self.call("/history",token=token)[1];self.assertEqual(len(h),1)
        self.assertEqual(h[0]["input"],sample)
        # Provision roles directly only in this isolated test database.
        from sqlalchemy import create_engine, text
        engine=create_engine(self.url_db)
        with engine.begin() as conn:
            conn.execute(text("UPDATE users SET role='admin' WHERE email='test@example.com'"))
        self.assertEqual(self.call("/admin/users",token=token)[0],200)
        self.assertEqual(self.call("/admin/users/1","PUT",{"role":"user"},token)[0],400)
        self.assertEqual(self.call("/admin/users/1","DELETE",token=token)[0],400)
        self.assertEqual(self.call("/admin/system",token=token)[1]["threshold"],pred["threshold"])
        self.assertEqual(self.call("/admin/model/reload","POST",token=token)[0],200)
        second=dict(register,email="patient@example.com")
        self.assertEqual(self.call("/auth/register","POST",second)[0],201)
        second_token=self.call("/auth/login","POST",second)[1]["access_token"]
        self.assertEqual(self.call("/history",token=second_token)[1],[])
        self.assertEqual(self.call("/admin/users/2","PUT",{"is_active":False},token)[0],200)
        self.assertEqual(self.call("/history",token=second_token)[0],403)
        self.assertEqual(self.call("/auth/login","POST",second)[0],403)
        self.assertEqual(self.call("/admin/users/2","PUT",{"is_active":True},token)[0],200)
        with engine.begin() as conn:
            conn.execute(text("UPDATE users SET role='doctor' WHERE id=1"))
        self.assertEqual(self.call("/doctor/patients",token=token)[0],200)
        code,created=self.call("/doctor/patients","POST",{"full_name":"Demo Patient", "region":"Ha Noi"},token)
        self.assertEqual(code,201,created)
        code,result=self.call("/predict?patient_id="+str(created["id"])+"&cutoff=0.25","POST",sample,token)
        self.assertEqual(code,200,result); self.assertEqual(result["threshold"],.25)
        self.assertEqual(len(self.call('/doctor/patient/'+str(created['id'])+'/history',token=token)[1]),1)
        self.assertEqual(self.call('/predict?cutoff=.2','POST',sample,second_token)[0],403)
        self.assertEqual(self.call('/doctor/template',token=token)[0],200)
        self.assertEqual(self.call('/auth/demo/admin','POST')[0],200)
        admin_token=self.call('/auth/demo/admin','POST')[1]['access_token']
        self.assertEqual(self.call('/admin/stats',token=admin_token)[0],200)
        self.assertTrue(self.call('/admin/history',token=admin_token)[1])
        # Verify sensitive columns are encrypted in the physical database.
        with engine.connect() as conn:
            self.assertTrue(conn.execute(text("SELECT action FROM history LIMIT 1")).scalar().startswith('enc:v1:'))
            self.assertTrue(conn.execute(text("SELECT full_name FROM users LIMIT 1")).scalar().startswith('enc:v1:'))
        # Multipart batch test, one valid row and one invalid row.
        boundary='StudentBatchBoundary'
        csv='full_name,age,gender,hypertension,heart_disease,ever_married,work_type,Residence_type,avg_glucose_level,bmi,smoking_status\nDemo Batch,60,Male,1,0,Yes,Private,Urban,150,25,never smoked\nInvalid,200,Male,0,0,Yes,Private,Urban,100,24,Unknown\n'
        raw=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="patients.csv"\r\nContent-Type: text/csv\r\n\r\n'+csv+f'\r\n--{boundary}--\r\n').encode()
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}/doctor/batch',data=raw,headers={'Authorization':'Bearer '+token,'Content-Type':'multipart/form-data; boundary='+boundary},method='POST')
        with urllib.request.urlopen(req) as response:
            result=json.load(response)
            self.assertEqual(result['success'],1);self.assertEqual(result['total'],2)

        # XLSX input: exercise shared-free inline strings and the import reader.
        from xml.sax.saxutils import escape
        columns=['full_name']+list(sample)
        cells=[['Excel Demo']+list(sample.values())]
        xml_rows=[]
        for n,row in enumerate([columns]+cells,1):
            xml_rows.append('<row r="'+str(n)+'">'+''.join('<c r="'+chr(65+i)+str(n)+'" t="inlineStr"><is><t>'+escape(str(v))+'</t></is></c>' for i,v in enumerate(row))+'</row>')
        buffer=io.BytesIO()
        with zipfile.ZipFile(buffer,'w') as z:
            z.writestr('xl/worksheets/sheet1.xml','<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main"><sheetData>'+''.join(xml_rows)+'</sheetData></worksheet>')
        raw=(f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="patients.xlsx"\r\nContent-Type: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet\r\n\r\n').encode()+buffer.getvalue()+f'\r\n--{boundary}--\r\n'.encode()
        req=urllib.request.Request(f'http://127.0.0.1:{self.port}/doctor/batch',data=raw,headers={'Authorization':'Bearer '+token,'Content-Type':'multipart/form-data; boundary='+boundary},method='POST')
        with urllib.request.urlopen(req) as response:self.assertEqual(json.load(response)['success'],1)
        self.assertEqual(self.call("/doctor/patient/2/history",token=token)[1],[])
        status,content=self.call("/doctor/export",token=token)
        self.assertEqual(status,200)
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            self.assertIn("xl/worksheets/sheet1.xml",z.namelist())
        status,pdf=self.call("/doctor/export?format=pdf",token=token)
        self.assertEqual(status,200); self.assertTrue(pdf.startswith(b"%PDF-1.4"))
        self.assertEqual(self.call("/doctor/export?format=bad",token=token)[0],422)
        engine.dispose()

if __name__ == "__main__": unittest.main(verbosity=2)
