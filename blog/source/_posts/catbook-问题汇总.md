---
title: catbook bug汇总
date: 2023-06-14 00:55:49
categories:
- 项目 
tags:
- catbook
- bug
---

### OAuth 2.0 Client ID认证
#### ngrok
```bash
npm install -g ngrok
ngrok http 5050 --host-header="localhost:5050"
```
> 参考资料：
> 1. [Setting up OAuth 2.0 - Google Cloud Platform Console Help](https://support.google.com/cloud/answer/6158849#authorized-domains&zippy=%2Cauthorized-domains%2Cstep-configure-your-app-to-use-the-new-secret%2Cuser-consent%2Cweb-applications)
> 2. [localhost - OAuth: how to test with local URLs? - Stack Overflow](https://stackoverflow.com/questions/10456174/oauth-how-to-test-with-local-urls)
> 3. [Testing Google OAuth 2.0 with localhost? - Stack Overflow](https://stackoverflow.com/questions/56436510/testing-google-oauth-2-0-with-localhost)
> 4. [How to Test App Authentication Locally with ngrok | by Karen White | BigCommerce Developer Blog | Medium](https://medium.com/bigcommerce-developer-blog/how-to-test-app-authentication-locally-with-ngrok-149150bfe4cf)
> 5. [reactjs - Invalid Host Header when ngrok tries to connect to React dev server - Stack Overflow](https://stackoverflow.com/questions/45425721/invalid-host-header-when-ngrok-tries-to-connect-to-react-dev-server) 
> 6. [google-auth-library - npm](https://www.npmjs.com/package/google-auth-library)
> 7. [Google Identity Services Login with React (2023 React Google Login) - YouTube](https://www.youtube.com/watch?v=roxC8SMs7HU&ab_channel=CooperCodes)
> 8. [从 Google 登录服务迁移  |  Authentication  |  Google for Developers](https://developers.google.com/identity/gsi/web/guides/migration?hl=zh-cn#redirect-mode_1)
> 9. [@react-oauth/google - npm](https://www.npmjs.com/package/@react-oauth/google)
> 10. [Google Login Integration with React and npm || How to add Google login button in website || #google - YouTube](https://www.youtube.com/watch?v=n3P55o3Gfy0)
