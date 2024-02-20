/* eslint-disable jsx-a11y/anchor-is-valid */
import React from "react";
import "./Login.css";
import axios from "axios";
import { useState } from "react";
import { Link } from "react-router-dom";

// import {
//   TextField
// } from "@mui/material";

export default function Login() {
  localStorage.clear();
  const [user, setUser] = useState("");
  const [pass, setPass] = useState("");

  // const [loginForm, setloginForm] = useState({
  //     user: "",
  //     password: ""
  //   })

  const handleSubmit = async (e) => {
    // e.preventDefault();
    //   console.log(user+" "+pass)
    const data = { username: user, password: pass };
    console.log(data);
    try {
      const res = await axios.post("http://127.0.0.1:3001/login", data);

      console.log(res)
      if (res.data.ok === true) {
        localStorage.setItem("User", user);
        window.location = "/dashboard";
      } else {
        alert("Login Failed!!")
        localStorage.clear();
      }
    } catch (e) {
      alert(e);
    }
  };

  //   function handleChange(event) {
  //     const {value, name} = event.target
  //     setloginForm(prevNote => ({
  //         ...prevNote, [name]: value})
  //     )}

  return (
    <>
      <div className="form-size">
        <nav className="navbar navbar-expand-lg navbar-light fixed-top">
          <div className="container">
            <Link className="navbar-brand" to={"/sign-in"}>
              FashioNet
            </Link>
            <div className="collapse navbar-collapse" id="navbarTogglerDemo02">
              <ul className="navbar-nav ml-auto">
                <li className="nav-item">
                  <Link className="nav-link" to={"/"}>
                    Login
                  </Link>
                </li>
                <li className="nav-item">
                  <Link className="nav-link" to={"/sign-up"}>
                    Sign up
                  </Link>
                </li>
              </ul>
            </div>
          </div>
        </nav>
        <form>
          <h3>Sign In</h3>

          <div className="mb-3">
            <label>Username</label>
            <input
              // name="user"
              // text={loginForm.user}
              type="text"
              className="form-control"
              placeholder="Enter username"
              value={user}
              onChange={(e) => setUser(e.target.value)}
            />
          </div>
          <div className="mb-3">
            <label>Password</label>
            <input
              type="password"
              className="form-control"
              placeholder="Enter password"
              value={pass}
              onChange={(e) => setPass(e.target.value)}
            />
          </div>

          <div className="d-grid">
            <button
              type="button"
              onClick={() => handleSubmit()}
              className="btn btn-primary"
            >
              Submit
            </button>
          </div>
        </form>
      </div>
    </>
  );
}
