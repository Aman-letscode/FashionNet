import React from "react";
// import { Select,Option,MenuItem } from "@mui/base";
import './Login.css';
import axios from "axios";
import { useState } from "react";



const options = [
    {value:"man", label:"Male"},
    {value:"woman", label:"Female"}
]
export default function SignUp(){
        
        localStorage.clear();
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");
  const [email, setEmail] = useState("");
  const [gender, setGender] = useState("man");

  const handleSubmit = async () => {
    // e.preventDefault();
    const data = { username: user, password: pass, email: email,gender:gender };
    console.log(data)
    try {
      const res = await axios.post("http://localhost:3001/signup", data);

      if (res.data.ok === true) {       
            alert("Signed Up Successfully!!");
          window.location = "/";
        }
      
      else{
        alert("Sign Up Unsuccessfully!!");
        localStorage.clear();
        window.location = "/signup"
      }
    } catch (e) {
      alert(e);
    }
  };
        return (
            <>
            <div className="form-size">
            
            <form>
                <h3>Sign Up</h3>

                <div className="mb-3">
                    <label>Username</label>
                    <input
                        type="text"
                        className="form-control"
                        placeholder="First name"
                        value={user}
                            onChange={(e)=>setUser(e.target.value)}
                    />
                </div>
                <div className="mb-3">
                    <label>Email address</label>
                    <input
                        type="email"
                        className="form-control"
                        placeholder="Enter email"
                        value={email}
                        onChange={(e)=>setEmail(e.target.value)}
                    />
                </div>
                <div className="mb-3">
                    <label>Password</label>
                    <input
                        type="password"
                        className="form-control"
                        placeholder="Enter password"
                        value={pass}
                        onChange={(e)=>setPass(e.target.value)}
                    />
                </div>
                <div className="mb-3">
                <label>Gender</label>
                <select
                className="mb-1"
        value={gender}
        onChange={(e)=>setGender(e.target.value)}
        style={{marginLeft:"1rem",borderColor:"#dee2e6",borderRadius:"6px", padding:"2px"}}
      >
        {options.map((option) => (
        //   <MenuItem   key={option.value} value={option.value}>
        //     {option.label}
        //   </MenuItem>
            <option value={option.value}>{option.label}</option>
        ))}
      </select>
                </div>
                <div className="d-grid">
                    <button type="button" onClick={()=>handleSubmit()} className="btn btn-primary">
                        Sign Up
                    </button>
                </div>
                <p className="forgot-password text-right">
                    Already registered <a href="/" style={{textDecorationLine:"unset"}}>sign in?</a>
                </p>
            </form>
            </div>
            </>
        )
    }
