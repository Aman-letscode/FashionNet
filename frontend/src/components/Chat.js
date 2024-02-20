import React, { useState, useContext, useEffect } from "react";
import SmartToyIcon from "@mui/icons-material/SmartToy";
import AccountCircleIcon from "@mui/icons-material/AccountCircle";
import axios from "axios";
import {
  TextField,
  Paper,
  Typography,
  List,
  ListItem,
  ListItemIcon,
  ListItemText,
  Divider,
  IconButton,
} from "@mui/material";
import SendIcon from "@mui/icons-material/Send";
import { UserContext } from "../context/UserContext";
import { urlToCall, urlToSuggest } from "../constants/constants";
import { conversations } from "../constants/constants";

const Chat = () => {
  const [messages, setMessages] = useState([]);
  const [inputValue, setInputValue] = useState("");
  const { currentUser } = useContext(UserContext);

  const [gender, setGender] = useState("");

  useEffect(() => console.log(messages, gender), [messages, gender]);
  const user = localStorage.getItem("User");
  const handleInputChange = (event) => {
    setInputValue(event.target.value);
  };

  const handleKeyPress = (event) => {
    if (event.key === "Enter") {
      handleSendMessage();
    }
  };

  const handleSendMessage = async () => {
    if (inputValue.trim() !== "") {
      try {
        const jsonToSend = {
          query: inputValue,
          user: user,
        };
        const result = await makePostRequest(urlToCall, jsonToSend);
        // const message = concatenateWithLineNumbers(result.answer);

        console.log(result);

        //TODO change this according to API return

        setGender(result.gender);
        let newText = result.answer.map((item, i) => (
          <div>
            {/* <a href="" target="_blank"> */}
            {i == 0 ? (
              <p key={i}>{item}</p>
            ) : (
              <IconButton
                color="primary"
                size="small"
                onClick={() => handleSendItem(item)}
                style={{ marginTop: "1rem" }}
              >
                {i}.{item}
              </IconButton>
            )}

            <Divider />
          </div>
        ));

        setMessages(prevMessages=>[
          ...prevMessages,
          { user: currentUser, message: inputValue, type: "question" },
          {
            user: currentUser,
            message: newText,
            gender: result.gender,
            type: "answer",
          },
        ]);
      } catch (error) {
        console.error("Error:", error.message);
      }
      setInputValue("");
    }
  };
  const handleSendItem = async (item) => {
    try {
      const data = {
        query: item,
        gender: gender,
      };

      const result1 = await makePostRequest(urlToSuggest, data);

      let newText = result1.answer.map((item, i) => (
        <div>
          <p key={i}>
            {i + 1}. {item.name}
          </p>
          <p key={i + 100}>
            <a href={item.url} target="_blank">
              {item.url}
            </a>
          </p>
          <p key={i + 1000}>Price: Rs.{item.price} /-</p>
          <Divider />
        </div>
      ));
      // console.log(messages[0].message);
      setMessages(prevMessages=>[
        ...prevMessages,
        { user: currentUser, message: inputValue, type: "question" },
        {
          user: currentUser,
          message: newText,
          type: "answer",
        },
      ]);
      // console.log(messages[0].message)
    } catch (error) {
      console.error("Error:", error.message);
    }
    setInputValue("");
  };
  const makePostRequest = async (url, data) => {
    try {
      const response = await axios.post(url, data);
      return response.data;
    } catch (error) {
      throw error;
    }
  };
  const handleLogout = () =>{
    window.location = "/"
  }
  return (
    <>
      <Paper
        elevation={3}
        style={{
          padding: "1rem",
          height: "90vh",
          display: "flex",
          flexDirection: "column",
        }}
      >
        <Typography variant="h4" align="center" gutterBottom>
          FashionNet
        <IconButton
            color="primary"
            size="small"
            onClick={handleLogout}
            style={{ left:"30%" }} align="right"
          >Logout</IconButton>
        </Typography>
        <Divider />
        <div style={{ flex: 1, overflowY: "auto" }}>
          <List>
            {messages
              .filter((message) => message.user === currentUser)
              .map((message, index) => {
                if (message.type === "question") {
                  return (
                    <React.Fragment key={index}>
                      <ListItem
                        alignItems="flex-start"
                        style={{
                          justifyContent: "flex-end",
                        }}
                        sx={{ mt: 2 }}
                      >
                        <ListItemText
                          primary={message.message}
                          style={{ textAlign: "right" }}
                        />
                        <ListItemIcon
                          sx={{ marginBottom: 1, marginLeft: 2, mt: 0 }}
                        >
                          <AccountCircleIcon />
                        </ListItemIcon>
                      </ListItem>
                    </React.Fragment>
                  );
                }
                return (
                  <React.Fragment key={index}>
                    {/* <img
                      src={message.thumbnailImageUrl}
                      alt="Recommended Product"
                    /> */}
                    <ListItem alignItems="flex-start">
                      <ListItemIcon
                        sx={{ marginBottom: 1, marginLeft: 2, mt: 0 }}
                      >
                        <SmartToyIcon />
                      </ListItemIcon>
                      <div style={{ textAlign: "left" }}>{message.message}</div>

                      {/* <ListItemText
                        primary={message.message}
                        style={{ textAlign: "left" }}
                      /> */}
                    </ListItem>
                    <Divider />
                  </React.Fragment>
                );
              })}
          </List>
        </div>

        <div style={{ display: "flex", alignItems: "center" }}>
          <TextField
            label="Enter Your Outfit Requirements"
            variant="outlined"
            value={inputValue}
            onChange={handleInputChange}
            onKeyDown={handleKeyPress}
            fullWidth
            style={{ marginTop: "1rem" }}
          />
          <IconButton
            color="primary"
            size="large"
            onClick={handleSendMessage}
            style={{ marginTop: "1rem" }}
          >
            <SendIcon />
          </IconButton>
        </div>
      </Paper>
    </>
  );
};

export default Chat;
